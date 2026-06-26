"""QI coarse-correction, deflation, and residual-equation utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp
from sfincs_jax.solvers.preconditioner_qi_basis import (
    RHS1QICoarseBasis,
    RHS1QICoarseBlockLayout,
    orthonormalize_rhs1_qi_coarse_basis,
)
from dataclasses import asdict
import jax
import numpy as np
import math
import os
import time
from sfincs_jax.solvers.preconditioner_qi_basis import (
    build_rhs1_xblock_global_coarse_basis,
    build_rhs1_xblock_global_coupling_load_basis,
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
            "operator_on_basis_shape": tuple(
                (int(v) for v in self.operator_on_basis_shape)
            ),
            "operator_on_basis_norm": float(self.operator_on_basis_norm),
            "coarse_operator_shape": tuple(
                (int(v) for v in self.coarse_operator_shape)
            ),
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
        return _blockschur_regularized_action_least_squares(
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
            "improvement_ratio": None
            if self.improvement_ratio is None
            else float(self.improvement_ratio),
            "metadata": self.metadata.to_dict(),
        }


def _blockschur_empty_matrix(n_rows: int, dtype: Any) -> ArrayLike:
    return jnp.zeros((int(n_rows), 0), dtype=dtype)


def _blockschur_append(
    columns: list[ArrayLike], labels: list[str], values: ArrayLike | None, label: str
) -> None:
    if values is not None:
        columns.append(jnp.asarray(values).reshape((-1,)))
        labels.append(str(label))


def _blockschur_block_offsets(layout: RHS1QICoarseBlockLayout) -> tuple[int, ...]:
    offsets = [0]
    for size in layout.block_sizes:
        offsets.append(offsets[-1] + int(size))
    return tuple(offsets)


def _blockschur_centered_unit_weights(
    values: Sequence[float],
) -> tuple[float, ...] | None:
    raw = tuple((float(value) for value in values))
    if not raw:
        return None
    mean = sum(raw) / float(len(raw))
    centered = tuple((value - mean for value in raw))
    scale = max((abs(value) for value in centered), default=0.0)
    if scale <= 0.0:
        return None
    return tuple((value / scale for value in centered))


def _blockschur_centered_power_weights(
    values: Sequence[float], degree: int
) -> tuple[float, ...] | None:
    linear = _blockschur_centered_unit_weights(values)
    if linear is None:
        return None
    powered = tuple((value ** int(degree) for value in linear))
    return _blockschur_centered_unit_weights(powered)


def _blockschur_block_weighted_constant(
    layout: RHS1QICoarseBlockLayout, weights: Sequence[float], dtype: Any
) -> ArrayLike | None:
    if len(weights) != len(layout.block_sizes):
        raise ValueError("weights must match block_sizes")
    if not any((float(value) != 0.0 for value in weights)):
        return None
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _blockschur_block_offsets(layout)
    for block_index, weight in enumerate(weights):
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(float(weight))
    return values


def _blockschur_weights_for_blocks(
    n_blocks: int, weighted_blocks: Sequence[tuple[int, float]]
) -> tuple[float, ...]:
    weights = [0.0 for _ in range(int(n_blocks))]
    for block_index, value in weighted_blocks:
        weights[int(block_index)] = float(value)
    return tuple(weights)


def _blockschur_angular_specs(
    layout: RHS1QICoarseBlockLayout, *, max_angular_mode: int, dtype: Any
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
            theta_phase = (
                2.0 * jnp.pi * float(mode) * theta_grid / float(layout.n_theta)
            )
            specs.append((f"theta_cos{mode}", jnp.cos(theta_phase)))
            specs.append((f"theta_sin{mode}", jnp.sin(theta_phase)))
        if int(layout.n_zeta) > 1:
            zeta_phase = 2.0 * jnp.pi * float(mode) * zeta_grid / float(layout.n_zeta)
            specs.append((f"zeta_cos{mode}", jnp.cos(zeta_phase)))
            specs.append((f"zeta_sin{mode}", jnp.sin(zeta_phase)))
        if int(layout.n_theta) > 1 and int(layout.n_zeta) > 1:
            theta_phase = (
                2.0 * jnp.pi * float(mode) * theta_grid / float(layout.n_theta)
            )
            zeta_phase = 2.0 * jnp.pi * float(mode) * zeta_grid / float(layout.n_zeta)
            specs.append((f"mixed_cos_plus{mode}", jnp.cos(theta_phase + zeta_phase)))
            specs.append((f"mixed_sin_plus{mode}", jnp.sin(theta_phase + zeta_phase)))
            specs.append((f"mixed_cos_minus{mode}", jnp.cos(theta_phase - zeta_phase)))
            specs.append((f"mixed_sin_minus{mode}", jnp.sin(theta_phase - zeta_phase)))
    return tuple(specs)


def _blockschur_angular_candidate(
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
    if weights is not None and (not any((float(value) != 0.0 for value in weights))):
        return None
    angular = jnp.asarray(angular_values, dtype=dtype).reshape((n_angular,))
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _blockschur_block_offsets(layout)
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
    block_x = tuple(
        (float(value) for value in layout.block_x or tuple(range(n_blocks)))
    )

    def has_room() -> bool:
        return len(columns) < limit

    def add(values: ArrayLike | None, label: str) -> None:
        if has_room():
            _blockschur_append(columns, labels, values, label)

    if limit <= 0:
        return (_blockschur_empty_matrix(layout.total_size, dtype), ())
    angular_specs = _blockschur_angular_specs(
        layout, max_angular_mode=max_angular_mode, dtype=dtype
    )
    radial_weights: list[tuple[str, tuple[float, ...]]] = []
    for degree in range(1, max(0, int(max_radial_degree)) + 1):
        weights = _blockschur_centered_power_weights(block_x, degree)
        if weights is not None:
            radial_weights.append((f"radial:p{degree}", weights))
    if include_global:
        add(jnp.ones((layout.total_size,), dtype=dtype), "global")
    if include_radial:
        for label, weights in radial_weights:
            add(_blockschur_block_weighted_constant(layout, weights, dtype), label)
    if include_angular:
        for label, values in angular_specs:
            add(
                _blockschur_angular_candidate(layout, values, None, dtype),
                f"angular:{label}",
            )
    if include_radial_angular:
        for radial_label, weights in radial_weights:
            for angular_label, values in angular_specs:
                add(
                    _blockschur_angular_candidate(layout, values, weights, dtype),
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
                weighted = tuple(
                    ((block, -1.0) for block in x_to_blocks[left_x])
                ) + tuple(((block, 1.0) for block in x_to_blocks[right_x]))
                weights = _blockschur_weights_for_blocks(n_blocks, weighted)
                schur_weights.append((f"schur:x_diff:{left_x}->{right_x}", weights))
                add(
                    _blockschur_block_weighted_constant(layout, weights, dtype),
                    schur_weights[-1][0],
                )
            for left_x, center_x, right_x in zip(
                x_values, x_values[1:], x_values[2:], strict=False
            ):
                weighted = (
                    tuple(((block, 1.0) for block in x_to_blocks[left_x]))
                    + tuple(((block, -2.0) for block in x_to_blocks[center_x]))
                    + tuple(((block, 1.0) for block in x_to_blocks[right_x]))
                )
                weights = _blockschur_weights_for_blocks(n_blocks, weighted)
                schur_weights.append(
                    (f"schur:x_curve:{left_x},{center_x},{right_x}", weights)
                )
                add(
                    _blockschur_block_weighted_constant(layout, weights, dtype),
                    schur_weights[-1][0],
                )
        if include_block_schur_angular:
            for schur_label, weights in schur_weights:
                for angular_label, values in angular_specs:
                    add(
                        _blockschur_angular_candidate(layout, values, weights, dtype),
                        f"{schur_label}*angular:{angular_label}",
                    )
    if include_blocks:
        for block_index in range(n_blocks):
            weights = _blockschur_weights_for_blocks(n_blocks, ((block_index, 1.0),))
            add(
                _blockschur_block_weighted_constant(layout, weights, dtype),
                f"block:{block_index}",
            )
    if not columns:
        return (_blockschur_empty_matrix(layout.total_size, dtype), ())
    return (jnp.stack(tuple(columns), axis=1), tuple(labels))


def build_rhs1_qi_block_schur_basis(
    layout: RHS1QICoarseBlockLayout,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_rank: int = 48,
    dtype: Any = jnp.float64,
    **candidate_options: Any,
) -> RHS1QICoarseBasis:
    """Build and rank-gate the block-Schur/angular/radial coarse basis."""
    candidates, labels = build_rhs1_qi_block_schur_candidates(
        layout, dtype=dtype, **candidate_options
    )
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates, labels=labels, rtol=rtol, atol=atol, max_rank=max(0, int(max_rank))
    )


def _blockschur_apply_operator_to_basis(
    operator: LinearOperator, basis_vectors: ArrayLike
) -> ArrayLike:
    q = jnp.asarray(basis_vectors)
    if int(q.shape[1]) == 0:
        return _blockschur_empty_matrix(int(q.shape[0]), q.dtype)
    return jnp.stack(
        tuple(
            (
                jnp.asarray(operator(q[:, i])).reshape((-1,))
                for i in range(int(q.shape[1]))
            )
        ),
        axis=1,
    )


def _blockschur_regularized_action_least_squares(
    matrix: ArrayLike, rhs: ArrayLike, *, rcond: float
) -> ArrayLike:
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


def _blockschur_conditioning_metadata(
    operator_on_basis: ArrayLike, *, rcond: float, atol: float
) -> tuple[int, float, float, float, float, float]:
    aq = jnp.asarray(operator_on_basis)
    if aq.size == 0 or int(aq.shape[1]) == 0:
        return (0, 0.0, 0.0, 0.0, 0.0, 0.0)
    singular_values = jnp.linalg.svd(aq, compute_uv=False)
    max_sv = float(jnp.max(singular_values))
    min_sv = float(jnp.min(singular_values))
    threshold = max(float(atol), max_sv * max(0.0, float(rcond)))
    numerical_rank = int(jnp.sum(singular_values > threshold))
    frob_sq = float(jnp.sum(singular_values * singular_values))
    stable_rank = 0.0 if max_sv <= 0.0 else frob_sq / (max_sv * max_sv)
    denom = max(min_sv, threshold, 1e-300)
    condition = 0.0 if max_sv <= 0.0 else max_sv / denom
    return (numerical_rank, stable_rank, condition, min_sv, max_sv, threshold)


def build_rhs1_qi_block_schur_preconditioner(
    *,
    operator: LinearOperator,
    layout: RHS1QICoarseBlockLayout | None = None,
    basis: RHS1QICoarseBasis | None = None,
    local_smoother: LinearOperator | None = None,
    regularization_rcond: float = 1e-12,
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
    smoother = (
        (lambda residual: jnp.zeros_like(jnp.asarray(residual).reshape((-1,))))
        if local_smoother is None
        else local_smoother
    )
    rank = int(q.shape[1])
    if rank <= 0:
        operator_on_basis = _blockschur_empty_matrix(int(q.shape[0]), q.dtype)
        coarse_operator = jnp.zeros((0, 0), dtype=q.dtype)
    else:
        operator_on_basis = _blockschur_apply_operator_to_basis(operator, q)
        coarse_operator = jnp.conjugate(q).T @ operator_on_basis
    numerical_rank, stable_rank, condition, min_sv, max_sv, threshold = (
        _blockschur_conditioning_metadata(
            operator_on_basis,
            rcond=float(regularization_rcond),
            atol=float(conditioning_atol),
        )
    )
    reason = "built" if rank > 0 and numerical_rank > 0 else "empty_or_rank_deficient"
    metadata = RHS1QIBlockSchurMetadata(
        total_size=int(q.shape[0]),
        candidate_count=int(basis.metadata.candidate_count),
        rank=rank,
        numerical_rank=int(numerical_rank),
        discarded_count=int(basis.metadata.discarded_count),
        operator_on_basis_shape=tuple((int(v) for v in operator_on_basis.shape)),
        operator_on_basis_norm=float(jnp.linalg.norm(operator_on_basis))
        if rank > 0
        else 0.0,
        coarse_operator_shape=tuple((int(v) for v in coarse_operator.shape)),
        coarse_operator_norm=float(jnp.linalg.norm(coarse_operator))
        if rank > 0
        else 0.0,
        stable_rank=float(stable_rank),
        condition_estimate=float(condition),
        min_singular_value=float(min_sv),
        max_singular_value=float(max_sv),
        singular_threshold=float(threshold),
        regularization_rcond=float(regularization_rcond),
        damping=float(damping),
        accepted_labels=tuple((str(label) for label in basis.metadata.accepted_labels)),
        candidate_labels=tuple(
            (str(label) for label in basis.metadata.candidate_labels)
        ),
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
        return (x_initial, probe)
    if int(preconditioner.metadata.rank) <= 0:
        probe = RHS1QIBlockSchurProbe(
            accepted=False,
            reason="empty_basis",
            residual_before_norm=residual_before_norm,
            residual_after_norm=residual_before_norm,
            improvement_ratio=1.0,
            metadata=preconditioner.metadata,
        )
        return (x_initial, probe)
    dx = jnp.asarray(preconditioner.apply(residual_before)).reshape((-1,))
    candidate = x_initial + dx
    residual_after = rhs_vec - jnp.asarray(operator(candidate)).reshape((-1,))
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = residual_after_norm / residual_before_norm
    required_drop = max(
        float(acceptance_atol),
        residual_before_norm * max(0.0, float(min_relative_improvement)),
    )
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
    return (candidate if accepted else x_initial, probe)


@dataclass(frozen=True)
class RHS1QICoupledResidualEquationConfig:
    """Static controls for the joint coarse residual equation."""

    max_rank: int | None = 96
    solver: str = "action_lstsq"
    regularization_rcond: float = 1e-12
    rank_rtol: float = 1e-10
    rank_atol: float = 0.0
    min_relative_improvement: float = 0.0
    acceptance_atol: float = 0.0


@dataclass(frozen=True)
class RHS1QICoupledResidualEquationMetadata:
    """JSON-friendly setup diagnostics for a coupled residual stage."""

    candidate_count: int
    rank: int
    source_stage_count: int
    source_stage_ranks: tuple[int, ...]
    residual_before: float
    residual_after: float
    solver: str
    condition_estimate: float
    accepted: bool
    reason: str
    accepted_labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return plain Python containers suitable for trace metadata."""
        return asdict(self)


@dataclass(frozen=True)
class RHS1QICoupledResidualEquationState:
    """Reusable coupled residual-equation stage.

    ``basis`` stores the accepted combined basis ``Q`` and ``operator_on_basis``
    stores the cached action ``A Q``.  Applying the stage is a small dense JAX
    solve followed by a matrix-vector multiply, so the result can be closed over
    by JIT-compiled Krylov code.
    """

    basis: RHS1QICoarseBasis
    operator_on_basis: ArrayLike
    coarse_operator: ArrayLike
    metadata: RHS1QICoupledResidualEquationMetadata
    regularization_rcond: float
    solver: str

    def solve_coefficients(self, residual: ArrayLike) -> ArrayLike:
        """Solve the cached joint coarse problem for ``residual``."""
        residual_vec = jnp.asarray(residual, dtype=self.basis.vectors.dtype).reshape(
            (-1,)
        )
        if int(self.metadata.rank) <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        if self.solver == "galerkin":
            projected = jnp.conjugate(self.basis.vectors).T @ residual_vec
            return _coupledresidual_regularized_square_solve(
                self.coarse_operator, projected, rcond=float(self.regularization_rcond)
            )
        return _coupledresidual_regularized_action_least_squares(
            self.operator_on_basis, residual_vec, rcond=float(self.regularization_rcond)
        )

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Return the lifted coupled coarse correction for ``residual``."""
        residual_vec = jnp.asarray(residual, dtype=self.basis.vectors.dtype).reshape(
            (-1,)
        )
        if int(self.metadata.rank) <= 0:
            return jnp.zeros_like(residual_vec)
        return self.basis.vectors @ self.solve_coefficients(residual_vec)

    def residual_after_apply(self, residual: ArrayLike) -> ArrayLike:
        """Return ``residual - A @ apply(residual)`` using cached ``A Q``."""
        residual_vec = jnp.asarray(residual, dtype=self.basis.vectors.dtype).reshape(
            (-1,)
        )
        if int(self.metadata.rank) <= 0:
            return residual_vec
        coefficients = self.solve_coefficients(residual_vec)
        return residual_vec - self.operator_on_basis @ coefficients


def setup_rhs1_qi_coupled_residual_equation(
    *,
    operator: LinearOperator,
    residual: ArrayLike,
    bases: Sequence[RHS1QICoarseBasis],
    config: RHS1QICoupledResidualEquationConfig | None = None,
) -> RHS1QICoupledResidualEquationState:
    """Build a fail-closed joint coarse residual-equation stage."""
    cfg = config or RHS1QICoupledResidualEquationConfig()
    solver = _coupledresidual_normalize_solver(cfg.solver)
    residual_vec = jnp.asarray(residual).reshape((-1,))
    total_size = int(residual_vec.shape[0])
    source_stage_ranks = tuple((int(basis.metadata.rank) for basis in bases))
    residual_before = float(jnp.linalg.norm(residual_vec))
    empty = _coupledresidual_empty_state(
        total_size=total_size,
        dtype=residual_vec.dtype,
        cfg=cfg,
        solver=solver,
        source_stage_ranks=source_stage_ranks,
        residual_before=residual_before,
        reason="zero_residual" if residual_before == 0.0 else "empty_basis",
    )
    if residual_before == 0.0:
        return empty
    candidates, labels = _coupledresidual_collect_candidates(
        bases, total_size=total_size, dtype=residual_vec.dtype
    )
    if int(candidates.shape[1]) == 0:
        return empty
    max_rank = None if cfg.max_rank is None else max(0, int(cfg.max_rank))
    basis = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=float(cfg.rank_rtol),
        atol=float(cfg.rank_atol),
        max_rank=max_rank,
    )
    rank = int(basis.metadata.rank)
    if rank <= 0:
        return _coupledresidual_empty_state(
            total_size=total_size,
            dtype=residual_vec.dtype,
            cfg=cfg,
            solver=solver,
            source_stage_ranks=source_stage_ranks,
            residual_before=residual_before,
            reason="rank_deficient",
            candidate_labels=labels,
            candidate_count=int(candidates.shape[1]),
        )
    action = _coupledresidual_apply_operator_to_basis(operator, basis.vectors)
    coarse_operator = jnp.conjugate(basis.vectors).T @ action
    coefficients = _coupledresidual_solve_cached(
        basis=basis.vectors,
        action=action,
        coarse_operator=coarse_operator,
        residual=residual_vec,
        solver=solver,
        rcond=float(cfg.regularization_rcond),
    )
    residual_after_vec = residual_vec - action @ coefficients
    residual_after = float(jnp.linalg.norm(residual_after_vec))
    required_after = residual_before * (
        1.0 - max(0.0, float(cfg.min_relative_improvement))
    )
    accepted = bool(residual_after <= required_after - float(cfg.acceptance_atol))
    condition_matrix = (
        coarse_operator if solver == "galerkin" else jnp.conjugate(action).T @ action
    )
    condition_estimate = _coupledresidual_condition_estimate(condition_matrix)
    reason = "accepted" if accepted else "no_residual_reduction"
    metadata = RHS1QICoupledResidualEquationMetadata(
        candidate_count=int(candidates.shape[1]),
        rank=rank if accepted else 0,
        source_stage_count=len(source_stage_ranks),
        source_stage_ranks=source_stage_ranks,
        residual_before=residual_before,
        residual_after=residual_after,
        solver=solver,
        condition_estimate=condition_estimate,
        accepted=accepted,
        reason=reason,
        accepted_labels=tuple((str(label) for label in basis.metadata.accepted_labels))
        if accepted
        else (),
        candidate_labels=labels,
    )
    if accepted:
        return RHS1QICoupledResidualEquationState(
            basis=basis,
            operator_on_basis=action,
            coarse_operator=coarse_operator,
            metadata=metadata,
            regularization_rcond=float(cfg.regularization_rcond),
            solver=solver,
        )
    rejected_basis = _coupledresidual_empty_basis(
        total_size, residual_vec.dtype, labels=labels, cfg=cfg
    )
    return RHS1QICoupledResidualEquationState(
        basis=rejected_basis,
        operator_on_basis=jnp.zeros((total_size, 0), dtype=residual_vec.dtype),
        coarse_operator=jnp.zeros((0, 0), dtype=residual_vec.dtype),
        metadata=metadata,
        regularization_rcond=float(cfg.regularization_rcond),
        solver=solver,
    )


def _coupledresidual_normalize_solver(value: str) -> str:
    solver = str(value).strip().lower().replace("-", "_")
    if solver in {"action", "action_ls", "least_squares", "lstsq", "staged"}:
        return "action_lstsq"
    if solver in {"galerkin", "projected", "qtaq", "coarse_grid", "schur"}:
        return "galerkin"
    if solver == "action_lstsq":
        return "action_lstsq"
    raise ValueError("solver must be 'action_lstsq' or 'galerkin'")


def _coupledresidual_collect_candidates(
    bases: Sequence[RHS1QICoarseBasis], *, total_size: int, dtype: Any
) -> tuple[ArrayLike, tuple[str, ...]]:
    columns: list[ArrayLike] = []
    labels: list[str] = []
    for stage_index, basis in enumerate(bases):
        vectors = jnp.asarray(basis.vectors, dtype=dtype)
        if vectors.ndim != 2:
            raise ValueError("basis vectors must be two-dimensional")
        if int(vectors.shape[0]) != int(total_size):
            raise ValueError("all bases must have the same row count as residual")
        for column_index, label in enumerate(basis.metadata.accepted_labels):
            columns.append(vectors[:, int(column_index)])
            labels.append(f"coupled:stage{stage_index}:{label}")
    if not columns:
        return (jnp.zeros((int(total_size), 0), dtype=dtype), ())
    return (jnp.stack(tuple(columns), axis=1), tuple(labels))


def _coupledresidual_apply_operator_to_basis(
    operator: LinearOperator, basis_vectors: ArrayLike
) -> ArrayLike:
    q = jnp.asarray(basis_vectors)
    if q.ndim != 2:
        raise ValueError("basis vectors must be a 2D matrix")
    if int(q.shape[1]) == 0:
        return jnp.zeros((int(q.shape[0]), 0), dtype=q.dtype)

    def apply_column(column: ArrayLike) -> ArrayLike:
        return jnp.asarray(operator(column), dtype=q.dtype).reshape((-1,))

    try:
        return jax.vmap(apply_column, in_axes=1, out_axes=1)(q)
    except NotImplementedError:
        return jnp.stack(
            tuple((apply_column(q[:, index]) for index in range(int(q.shape[1])))),
            axis=1,
        )


def _coupledresidual_solve_cached(
    *,
    basis: ArrayLike,
    action: ArrayLike,
    coarse_operator: ArrayLike,
    residual: ArrayLike,
    solver: str,
    rcond: float,
) -> ArrayLike:
    if solver == "galerkin":
        projected = jnp.conjugate(basis).T @ residual
        return _coupledresidual_regularized_square_solve(
            coarse_operator, projected, rcond=rcond
        )
    return _coupledresidual_regularized_action_least_squares(
        action, residual, rcond=rcond
    )


def _coupledresidual_regularized_action_least_squares(
    matrix: ArrayLike, rhs: ArrayLike, *, rcond: float
) -> ArrayLike:
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    gram = jnp.conjugate(a).T @ a
    normal_rhs = jnp.conjugate(a).T @ rhs_vec
    return _coupledresidual_regularized_square_solve(gram, normal_rhs, rcond=rcond)


def _coupledresidual_regularized_square_solve(
    matrix: ArrayLike, rhs: ArrayLike, *, rcond: float
) -> ArrayLike:
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if int(a.shape[0]) == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    row_sums = jnp.sum(jnp.abs(a), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=a.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=a.dtype) * scale
    eye = jnp.eye(int(a.shape[0]), dtype=a.dtype)
    return jnp.linalg.solve(a + ridge * eye, rhs_vec)


def _coupledresidual_condition_estimate(matrix: ArrayLike) -> float:
    a = jnp.asarray(matrix)
    if a.ndim != 2 or int(a.shape[0]) == 0 or int(a.shape[1]) == 0:
        return float("inf")
    try:
        return float(jnp.linalg.cond(a))
    except Exception:
        return float("inf")


def _coupledresidual_empty_basis(
    total_size: int,
    dtype: Any,
    *,
    labels: Sequence[str],
    cfg: RHS1QICoupledResidualEquationConfig,
) -> RHS1QICoarseBasis:
    return orthonormalize_rhs1_qi_coarse_basis(
        jnp.zeros((int(total_size), 0), dtype=dtype),
        labels=(),
        rtol=float(cfg.rank_rtol),
        atol=float(cfg.rank_atol),
        max_rank=0,
    )


def _coupledresidual_empty_state(
    *,
    total_size: int,
    dtype: Any,
    cfg: RHS1QICoupledResidualEquationConfig,
    solver: str,
    source_stage_ranks: tuple[int, ...],
    residual_before: float,
    reason: str,
    candidate_labels: Sequence[str] = (),
    candidate_count: int = 0,
) -> RHS1QICoupledResidualEquationState:
    basis = _coupledresidual_empty_basis(
        total_size, dtype, labels=candidate_labels, cfg=cfg
    )
    metadata = RHS1QICoupledResidualEquationMetadata(
        candidate_count=int(candidate_count),
        rank=0,
        source_stage_count=len(source_stage_ranks),
        source_stage_ranks=source_stage_ranks,
        residual_before=float(residual_before),
        residual_after=float(residual_before),
        solver=solver,
        condition_estimate=float("inf"),
        accepted=False,
        reason=reason,
        accepted_labels=(),
        candidate_labels=tuple((str(label) for label in candidate_labels)),
    )
    return RHS1QICoupledResidualEquationState(
        basis=basis,
        operator_on_basis=jnp.zeros((int(total_size), 0), dtype=dtype),
        coarse_operator=jnp.zeros((0, 0), dtype=dtype),
        metadata=metadata,
        regularization_rcond=float(cfg.regularization_rcond),
        solver=solver,
    )


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
            "basis_shape": tuple((int(v) for v in self.basis_shape)),
            "operator_on_basis_shape": tuple(
                (int(v) for v in self.operator_on_basis_shape)
            ),
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
        return _deflation_regularized_action_least_squares(
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
            coarse_input = residual_vec - jnp.asarray(self.operator(local)).reshape(
                (-1,)
            )
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
            "improvement_ratio": None
            if self.improvement_ratio is None
            else float(self.improvement_ratio),
            "metadata": self.metadata.to_dict(),
            "seed_solver": self.seed_solver,
            "cycle_residual_history": tuple(
                (float(v) for v in self.cycle_residual_history)
            ),
            "cycle_coefficients": tuple((float(v) for v in self.cycle_coefficients)),
        }


def _deflation_normalize_composition(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"add", "additive"}:
        return "additive"
    if normalized in {"mult", "multiplicative", "field_split", "schur"}:
        return "multiplicative"
    raise ValueError("composition must be 'additive' or 'multiplicative'")


def _deflation_append_candidate(
    columns: list[ArrayLike],
    labels: list[str],
    values: ArrayLike,
    label: str,
    *,
    total_size: int,
) -> None:
    vector = jnp.asarray(values).reshape((-1,))
    if int(vector.shape[0]) != int(total_size):
        raise ValueError(
            f"candidate {label!r} has length {vector.shape[0]}, expected {total_size}"
        )
    norm = float(jnp.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        return
    columns.append(vector / norm)
    labels.append(str(label))


def _deflation_apply_operator_to_basis(
    operator: LinearOperator, basis_vectors: ArrayLike
) -> ArrayLike:
    q = jnp.asarray(basis_vectors)
    if q.ndim != 2 or int(q.shape[1]) == 0:
        return jnp.zeros_like(q)
    columns = [
        jnp.asarray(operator(q[:, i])).reshape((-1,)) for i in range(int(q.shape[1]))
    ]
    return jnp.stack(columns, axis=1)


def _deflation_regularized_action_least_squares(
    a: ArrayLike, rhs: ArrayLike, *, rcond: float
) -> ArrayLike:
    a_arr = jnp.asarray(a)
    rhs_arr = jnp.asarray(rhs).reshape((-1,))
    if a_arr.ndim != 2:
        raise ValueError("least-squares operator must be a matrix")
    if int(a_arr.shape[1]) == 0:
        return jnp.zeros((0,), dtype=rhs_arr.dtype)
    gram = jnp.conjugate(a_arr).T @ a_arr
    normal_rhs = jnp.conjugate(a_arr).T @ rhs_arr
    scale = jnp.maximum(jnp.linalg.norm(gram), jnp.asarray(1.0, dtype=gram.dtype))
    ridge = jnp.asarray(max(float(rcond), 1e-14), dtype=gram.dtype) * scale
    eye = jnp.eye(int(gram.shape[0]), dtype=gram.dtype)
    return jnp.linalg.solve(gram + ridge * eye, normal_rhs)


def _deflation_singular_diagnostics(
    operator_on_basis: ArrayLike,
) -> tuple[float, float, float, float]:
    a = jnp.asarray(operator_on_basis)
    if a.ndim != 2 or int(a.shape[1]) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    singular_values = np.asarray(jnp.linalg.svd(a, compute_uv=False), dtype=np.float64)
    if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
        return (0.0, 0.0, 0.0, 0.0)
    max_sv = float(np.max(singular_values))
    min_sv = float(np.min(singular_values))
    frob_sq = float(np.sum(singular_values * singular_values))
    stable_rank = frob_sq / max(max_sv * max_sv, 1e-300)
    condition = max_sv / max(min_sv, 1e-300)
    return (min_sv, max_sv, stable_rank, condition)


def build_rhs1_qi_residual_deflated_preconditioner(
    *,
    operator: LinearOperator,
    local_smoother: LinearOperator,
    residual_seed: ArrayLike,
    extra_directions: Sequence[tuple[str, ArrayLike]] = (),
    krylov_depth: int = 4,
    max_rank: int = 16,
    regularization_rcond: float = 1e-12,
    basis_rtol: float = 1e-10,
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
    composition_use = _deflation_normalize_composition(composition)
    depth_use = max(0, int(krylov_depth))
    max_rank_use = max(0, int(max_rank))
    cycles_use = max(1, int(correction_cycles))
    columns: list[ArrayLike] = []
    labels: list[str] = []
    if bool(include_raw_residual):
        _deflation_append_candidate(
            columns, labels, residual, "residual", total_size=total_size
        )
    z = jnp.asarray(local_smoother(residual)).reshape((-1,))
    _deflation_append_candidate(
        columns, labels, z, "local_smoother(residual)", total_size=total_size
    )
    for depth in range(depth_use):
        az = jnp.asarray(operator(z)).reshape((-1,))
        z = jnp.asarray(local_smoother(az)).reshape((-1,))
        _deflation_append_candidate(
            columns,
            labels,
            z,
            f"preconditioned_krylov:{depth + 1}",
            total_size=total_size,
        )
    for label, direction in extra_directions:
        _deflation_append_candidate(
            columns, labels, direction, f"extra:{label}", total_size=total_size
        )
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
    operator_on_basis = _deflation_apply_operator_to_basis(operator, basis.vectors)
    op_norm = (
        float(jnp.linalg.norm(operator_on_basis))
        if int(basis.metadata.rank) > 0
        else 0.0
    )
    min_sv, max_sv, stable_rank, condition = _deflation_singular_diagnostics(
        operator_on_basis
    )
    metadata = RHS1QIDeflationMetadata(
        total_size=total_size,
        candidate_count=int(basis.metadata.candidate_count),
        rank=int(basis.metadata.rank),
        discarded_count=int(basis.metadata.discarded_count),
        requested_krylov_depth=depth_use,
        basis_shape=tuple((int(v) for v in basis.vectors.shape)),
        operator_on_basis_shape=tuple((int(v) for v in operator_on_basis.shape)),
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
        residual_after_norm / residual_before_norm
        if residual_before_norm > 0.0
        else None
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
    return (x_candidate if accepted else x_initial, probe)


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
        if (
            not np.isfinite(step_norm)
            or not np.isfinite(action_norm)
            or step_norm <= 0.0
            or (action_norm <= 0.0)
        ):
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
        return (x_initial, probe)
    z_basis = jnp.stack(tuple(correction_columns), axis=1)
    az_basis = jnp.stack(tuple(action_columns), axis=1)
    rcond = (
        float(preconditioner.metadata.regularization_rcond)
        if regularization_rcond is None
        else float(regularization_rcond)
    )
    coefficients = _deflation_regularized_action_least_squares(
        az_basis, residual_before, rcond=rcond
    )
    dx = z_basis @ coefficients
    x_candidate = x_initial + dx
    residual_after = rhs_vec - jnp.asarray(operator(x_candidate)).reshape((-1,))
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = (
        residual_after_norm / residual_before_norm
        if residual_before_norm > 0.0
        else None
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
        cycle_coefficients=tuple(
            (
                float(v)
                for v in np.asarray(coefficients, dtype=np.float64).reshape((-1,))
            )
        ),
    )
    return (x_candidate if accepted else x_initial, probe)


@dataclass(frozen=True)
class RHS1QIMultilevelCoarseConfig:
    """Static controls for angular-radial multilevel coarse construction."""

    max_levels: int = 3
    aggregate_factor: int = 2
    max_rank: int = 48
    max_angular_mode: int = 1
    max_radial_degree: int = 2
    include_level_aggregates: bool = True
    include_angular: bool = True
    include_radial: bool = True
    include_radial_angular: bool = True
    include_pitch: bool = True
    include_radial_pitch: bool = True
    include_current_moments: bool = False
    include_species_current_moments: bool = True
    include_radial_current_moments: bool = True
    include_tail_constraint_moments: bool = True
    include_finest_blocks: bool = False
    rtol: float = 1e-10
    atol: float = 0.0
    regularization_rcond: float = 1e-12
    damping: float = 1.0
    max_pitch_degree: int = 0
    max_current_pitch_degree: int = 1
    nested_residual_correction: bool = False
    nested_level_max_rank: int = 16
    nested_order: str = "coarse_to_fine"
    nested_solver: str = "action_lstsq"
    nested_include_global: bool = True
    dtype: Any = jnp.float64


@dataclass(frozen=True)
class RHS1QIMultilevelCoarseLevelMetadata:
    """Diagnostics for one radial aggregation level in the hierarchy."""

    level_index: int
    aggregate_size: int
    aggregate_count: int
    block_groups: tuple[tuple[int, ...], ...]
    candidate_count: int
    rank: int
    discarded_count: int
    candidate_labels: tuple[str, ...]
    accepted_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly per-level diagnostics."""
        return {
            "level_index": int(self.level_index),
            "aggregate_size": int(self.aggregate_size),
            "aggregate_count": int(self.aggregate_count),
            "block_groups": tuple(
                (tuple((int(v) for v in group)) for group in self.block_groups)
            ),
            "candidate_count": int(self.candidate_count),
            "rank": int(self.rank),
            "discarded_count": int(self.discarded_count),
            "candidate_labels": self.candidate_labels,
            "accepted_labels": self.accepted_labels,
        }


@dataclass(frozen=True)
class RHS1QIMultilevelCoarseMetadata:
    """Diagnostics for a reusable multilevel angular-radial coarse action."""

    total_size: int
    level_count: int
    levels: tuple[RHS1QIMultilevelCoarseLevelMetadata, ...]
    candidate_count: int
    rank: int
    nested_residual_correction_enabled: bool
    nested_level_count: int
    nested_rank: int
    nested_level_ranks: tuple[int, ...]
    nested_order: str
    nested_solver: str
    nested_include_global: bool
    nested_coarse_operator_shapes: tuple[tuple[int, int], ...]
    nested_coarse_operator_norms: tuple[float, ...]
    discarded_count: int
    operator_on_basis_shape: tuple[int, int]
    operator_on_basis_norm: float
    coarse_operator_shape: tuple[int, int]
    coarse_operator_norm: float
    regularization_rcond: float
    damping: float
    accepted_labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    device_resident: bool
    host_callback_free: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly diagnostics for traces and integration probes."""
        payload = asdict(self)
        payload["levels"] = tuple((level.to_dict() for level in self.levels))
        payload["operator_on_basis_shape"] = tuple(
            (int(v) for v in self.operator_on_basis_shape)
        )
        payload["coarse_operator_shape"] = tuple(
            (int(v) for v in self.coarse_operator_shape)
        )
        payload["nested_coarse_operator_shapes"] = tuple(
            (
                tuple((int(v) for v in shape))
                for shape in self.nested_coarse_operator_shapes
            )
        )
        return payload


@dataclass(frozen=True)
class RHS1QIMultilevelCoarsePreconditioner:
    """Reusable pure-JAX local-plus-multilevel-coarse correction."""

    operator: LinearOperator
    local_smoother: LinearOperator
    basis: RHS1QICoarseBasis
    operator_on_basis: ArrayLike
    coarse_operator: ArrayLike
    nested_bases: tuple[RHS1QICoarseBasis, ...]
    nested_operator_on_bases: tuple[ArrayLike, ...]
    nested_coarse_operators: tuple[ArrayLike, ...]
    metadata: RHS1QIMultilevelCoarseMetadata

    def solve_coarse(self, residual: ArrayLike) -> ArrayLike:
        """Return action least-squares coarse coefficients for ``residual``."""
        residual_vec = jnp.asarray(residual).reshape((-1,))
        rank = int(self.metadata.rank)
        if rank <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        return _multilevel_regularized_action_least_squares(
            self.operator_on_basis,
            residual_vec,
            rcond=float(self.metadata.regularization_rcond),
        )

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply one multiplicative local plus coarse correction."""
        residual_vec = jnp.asarray(residual).reshape((-1,))
        local = jnp.asarray(self.local_smoother(residual_vec)).reshape((-1,))
        if int(self.metadata.rank) <= 0 and int(self.metadata.nested_rank) <= 0:
            return float(self.metadata.damping) * local
        remaining = residual_vec - jnp.asarray(self.operator(local)).reshape((-1,))
        if bool(self.metadata.nested_residual_correction_enabled):
            coarse = self.solve_nested_residual_equation(remaining)
        else:
            coarse = self.basis.vectors @ self.solve_coarse(remaining)
        return float(self.metadata.damping) * (local + coarse)

    def solve_nested_residual_equation(self, residual: ArrayLike) -> ArrayLike:
        """Solve the coarse residual equation level by level.

        Each level receives a separate rank budget and acts on the residual left
        by previous levels. This is a bounded multilevel residual equation, not
        another smoother sweep: every stage minimizes ``||r - A Q_l c||`` over
        its own coarse action, updates by ``Q_l c``, and passes the remaining
        residual to the next coarser/finer level.
        """
        residual_vec = jnp.asarray(residual).reshape((-1,))
        correction = jnp.zeros_like(residual_vec)
        remaining = residual_vec
        solver = str(self.metadata.nested_solver)
        for basis, action, coarse_operator in zip(
            self.nested_bases,
            self.nested_operator_on_bases,
            self.nested_coarse_operators,
            strict=True,
        ):
            if int(basis.metadata.rank) <= 0:
                continue
            if solver == "galerkin":
                coefficients = _multilevel_regularized_galerkin_solve(
                    coarse_operator,
                    jnp.conjugate(
                        jnp.asarray(basis.vectors, dtype=residual_vec.dtype)
                    ).T
                    @ remaining,
                    rcond=float(self.metadata.regularization_rcond),
                )
            else:
                coefficients = _multilevel_regularized_action_least_squares(
                    action, remaining, rcond=float(self.metadata.regularization_rcond)
                )
            level_update = (
                jnp.asarray(basis.vectors, dtype=residual_vec.dtype) @ coefficients
            )
            correction = correction + level_update
            remaining = (
                remaining - jnp.asarray(action, dtype=residual_vec.dtype) @ coefficients
            )
        if bool(self.metadata.nested_include_global) and int(self.metadata.rank) > 0:
            if solver == "galerkin":
                coefficients = _multilevel_regularized_galerkin_solve(
                    self.coarse_operator,
                    jnp.conjugate(
                        jnp.asarray(self.basis.vectors, dtype=residual_vec.dtype)
                    ).T
                    @ remaining,
                    rcond=float(self.metadata.regularization_rcond),
                )
            else:
                coefficients = _multilevel_regularized_action_least_squares(
                    self.operator_on_basis,
                    remaining,
                    rcond=float(self.metadata.regularization_rcond),
                )
            correction = (
                correction
                + jnp.asarray(self.basis.vectors, dtype=residual_vec.dtype)
                @ coefficients
            )
        return correction

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for Krylov preconditioner hooks."""
        return self.apply


@dataclass(frozen=True)
class RHS1QIMultilevelCoarseProbe:
    """Fail-closed true-residual probe for the multilevel coarse candidate."""

    accepted: bool
    reason: str
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float | None
    metadata: RHS1QIMultilevelCoarseMetadata

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly probe diagnostics."""
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": float(self.residual_after_norm),
            "improvement_ratio": None
            if self.improvement_ratio is None
            else float(self.improvement_ratio),
            "metadata": self.metadata.to_dict(),
        }


def _multilevel_empty_matrix(n_rows: int, dtype: Any) -> ArrayLike:
    return jnp.zeros((int(n_rows), 0), dtype=dtype)


def _multilevel_block_offsets(layout: RHS1QICoarseBlockLayout) -> tuple[int, ...]:
    offsets = [0]
    for size in layout.block_sizes:
        offsets.append(offsets[-1] + int(size))
    return tuple(offsets)


def _multilevel_radial_groups(
    n_blocks: int, aggregate_size: int
) -> tuple[tuple[int, ...], ...]:
    size = max(1, int(aggregate_size))
    groups: list[tuple[int, ...]] = []
    for start in range(0, int(n_blocks), size):
        stop = min(int(n_blocks), start + size)
        groups.append(tuple(range(start, stop)))
    return tuple(groups)


def _multilevel_hierarchy_groups(
    n_blocks: int, config: RHS1QIMultilevelCoarseConfig
) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    max_levels = max(1, int(config.max_levels))
    factor = max(2, int(config.aggregate_factor))
    levels: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
    aggregate_size = 1
    previous_groups: tuple[tuple[int, ...], ...] | None = None
    for _ in range(max_levels):
        groups = _multilevel_radial_groups(n_blocks, aggregate_size)
        if groups == previous_groups:
            break
        levels.append((aggregate_size, groups))
        if len(groups) <= 1:
            break
        previous_groups = groups
        aggregate_size *= factor
    return tuple(levels)


def _multilevel_centered_unit_weights(
    values: Sequence[float],
) -> tuple[float, ...] | None:
    raw = tuple((float(value) for value in values))
    if not raw:
        return None
    mean = sum(raw) / float(len(raw))
    centered = tuple((value - mean for value in raw))
    scale = max((abs(value) for value in centered), default=0.0)
    if scale <= 0.0:
        return None
    return tuple((value / scale for value in centered))


def _multilevel_centered_power_weights(
    values: Sequence[float], degree: int
) -> tuple[float, ...] | None:
    linear = _multilevel_centered_unit_weights(values)
    if linear is None:
        return None
    if int(degree) == 1:
        return linear
    return _multilevel_centered_unit_weights(
        tuple((value ** int(degree) for value in linear))
    )


def _multilevel_angular_specs(
    layout: RHS1QICoarseBlockLayout, *, max_angular_mode: int, dtype: Any
) -> tuple[tuple[str, ArrayLike], ...]:
    n_theta = int(layout.n_theta)
    n_zeta = int(layout.n_zeta)
    max_mode = max(0, int(max_angular_mode))
    if max_mode <= 0 or n_theta * n_zeta <= 1:
        return ()
    theta = jnp.arange(n_theta, dtype=dtype)
    zeta = jnp.arange(n_zeta, dtype=dtype)
    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta, indexing="ij")
    specs: list[tuple[str, ArrayLike]] = []
    for mode in range(1, max_mode + 1):
        if n_theta > 1:
            theta_phase = 2.0 * jnp.pi * float(mode) * theta_grid / float(n_theta)
            specs.append((f"theta_cos{mode}", jnp.cos(theta_phase)))
            specs.append((f"theta_sin{mode}", jnp.sin(theta_phase)))
        if n_zeta > 1:
            zeta_phase = 2.0 * jnp.pi * float(mode) * zeta_grid / float(n_zeta)
            specs.append((f"zeta_cos{mode}", jnp.cos(zeta_phase)))
            specs.append((f"zeta_sin{mode}", jnp.sin(zeta_phase)))
        if n_theta > 1 and n_zeta > 1:
            theta_phase = 2.0 * jnp.pi * float(mode) * theta_grid / float(n_theta)
            zeta_phase = 2.0 * jnp.pi * float(mode) * zeta_grid / float(n_zeta)
            specs.append((f"mixed_cos_plus{mode}", jnp.cos(theta_phase + zeta_phase)))
            specs.append((f"mixed_sin_plus{mode}", jnp.sin(theta_phase + zeta_phase)))
            specs.append((f"mixed_cos_minus{mode}", jnp.cos(theta_phase - zeta_phase)))
            specs.append((f"mixed_sin_minus{mode}", jnp.sin(theta_phase - zeta_phase)))
    return tuple(specs)


def _multilevel_group_weighted_constant(
    layout: RHS1QICoarseBlockLayout,
    groups: Sequence[Sequence[int]],
    group_weights: Sequence[float],
    dtype: Any,
) -> ArrayLike | None:
    if len(groups) != len(group_weights):
        raise ValueError("group_weights must match groups")
    if not any((float(weight) != 0.0 for weight in group_weights)):
        return None
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _multilevel_block_offsets(layout)
    for group, weight in zip(groups, group_weights, strict=True):
        for block_index in group:
            start = int(offsets[int(block_index)])
            stop = int(offsets[int(block_index) + 1])
            values = values.at[start:stop].set(float(weight))
    return values


def _multilevel_block_weighted_constant(
    layout: RHS1QICoarseBlockLayout, block_weights: Sequence[float], dtype: Any
) -> ArrayLike | None:
    """Return a block-weighted constant over arbitrary structural blocks."""
    if len(block_weights) != len(layout.block_sizes):
        raise ValueError("block_weights must match block count")
    if not any((float(weight) != 0.0 for weight in block_weights)):
        return None
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _multilevel_block_offsets(layout)
    for block_index, weight in enumerate(block_weights):
        if float(weight) == 0.0:
            continue
        start = int(offsets[int(block_index)])
        stop = int(offsets[int(block_index) + 1])
        values = values.at[start:stop].set(float(weight))
    return values


def _multilevel_group_weighted_angular(
    layout: RHS1QICoarseBlockLayout,
    groups: Sequence[Sequence[int]],
    angular_values: ArrayLike,
    group_weights: Sequence[float],
    dtype: Any,
) -> ArrayLike | None:
    if len(groups) != len(group_weights):
        raise ValueError("group_weights must match groups")
    if not any((float(weight) != 0.0 for weight in group_weights)):
        return None
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 1:
        return None
    angular = jnp.asarray(angular_values, dtype=dtype).reshape((n_angular,))
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _multilevel_block_offsets(layout)
    for group, weight in zip(groups, group_weights, strict=True):
        for block_index in group:
            block_size = int(layout.block_sizes[int(block_index)])
            if block_size % n_angular != 0:
                return None
            repeats = block_size // n_angular
            start = int(offsets[int(block_index)])
            stop = int(offsets[int(block_index) + 1])
            values = values.at[start:stop].set(
                float(weight) * jnp.tile(angular, repeats)
            )
    return values


def _multilevel_pitch_weights(
    layout: RHS1QICoarseBlockLayout, *, degree: int
) -> tuple[float, ...] | None:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0:
        return None
    pitch_count: int | None = None
    for block_size in layout.block_sizes:
        if int(block_size) % n_angular != 0:
            return None
        block_pitch_count = int(block_size) // n_angular
        if block_pitch_count <= 1:
            return None
        if pitch_count is None:
            pitch_count = block_pitch_count
        elif pitch_count != block_pitch_count:
            return None
    if pitch_count is None:
        return None
    return _multilevel_centered_power_weights(tuple(range(pitch_count)), int(degree))


def _multilevel_pitch_weights_for_count(
    pitch_count: int, *, degree: int
) -> tuple[float, ...] | None:
    if int(pitch_count) <= 1:
        return None
    return _multilevel_centered_power_weights(
        tuple(range(int(pitch_count))), int(degree)
    )


def _multilevel_group_weighted_pitch(
    layout: RHS1QICoarseBlockLayout,
    groups: Sequence[Sequence[int]],
    pitch_weights: Sequence[float],
    group_weights: Sequence[float],
    dtype: Any,
) -> ArrayLike | None:
    """Return a group-weighted pitch moment candidate.

    SFINCS-JAX stores each block as repeated angular planes over pitch/xi.  A
    pitch moment is therefore the low-order xi polynomial repeated over each
    theta-zeta plane.  This adds a real coarse-space direction for PAS/QI slow
    modes that radial/angular aggregate spaces cannot represent.
    """
    if len(groups) != len(group_weights):
        raise ValueError("group_weights must match groups")
    if not any((float(weight) != 0.0 for weight in group_weights)):
        return None
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0:
        return None
    pitch = jnp.asarray(
        tuple((float(value) for value in pitch_weights)), dtype=dtype
    ).reshape((-1,))
    if int(pitch.shape[0]) <= 1:
        return None
    block_values = jnp.repeat(pitch, n_angular)
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _multilevel_block_offsets(layout)
    for group, weight in zip(groups, group_weights, strict=True):
        for block_index in group:
            block_size = int(layout.block_sizes[int(block_index)])
            if block_size != int(block_values.shape[0]):
                return None
            start = int(offsets[int(block_index)])
            stop = int(offsets[int(block_index) + 1])
            values = values.at[start:stop].set(float(weight) * block_values)
    return values


def _multilevel_block_weighted_pitch_moment(
    layout: RHS1QICoarseBlockLayout,
    block_weights: Sequence[float],
    *,
    degree: int,
    dtype: Any,
) -> ArrayLike | None:
    """Return a variable-pitch-count current/flow moment candidate.

    The production active-DOF layout can have a different number of retained xi
    points at different x locations.  Generic multilevel pitch candidates require
    identical block sizes, but current and bootstrap-current slow modes need a
    low-order xi moment even when the x=0 block is truncated.  This helper builds
    the centered pitch polynomial independently inside each supported block.
    """
    if len(block_weights) != len(layout.block_sizes):
        raise ValueError("block_weights must match block count")
    if not any((float(weight) != 0.0 for weight in block_weights)):
        return None
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0:
        return None
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _multilevel_block_offsets(layout)
    any_supported = False
    for block_index, weight in enumerate(block_weights):
        if float(weight) == 0.0:
            continue
        block_size = int(layout.block_sizes[int(block_index)])
        if block_size % n_angular != 0:
            continue
        pitch_count = block_size // n_angular
        pitch_weights = _multilevel_pitch_weights_for_count(
            pitch_count, degree=int(degree)
        )
        if pitch_weights is None:
            continue
        block_values = jnp.repeat(jnp.asarray(pitch_weights, dtype=dtype), n_angular)
        start = int(offsets[int(block_index)])
        stop = int(offsets[int(block_index) + 1])
        values = values.at[start:stop].set(float(weight) * block_values)
        any_supported = True
    return values if any_supported else None


def _multilevel_non_tail_block_indices(
    layout: RHS1QICoarseBlockLayout,
) -> tuple[int, ...]:
    result: list[int] = []
    for block_index, (x_value, species_value) in enumerate(
        zip(layout.block_x or (), layout.block_species or (), strict=True)
    ):
        if int(x_value) >= 0 and int(species_value) >= 0:
            result.append(int(block_index))
    return tuple(result)


def _multilevel_tail_block_indices(layout: RHS1QICoarseBlockLayout) -> tuple[int, ...]:
    result: list[int] = []
    for block_index, (x_value, species_value) in enumerate(
        zip(layout.block_x or (), layout.block_species or (), strict=True)
    ):
        if int(x_value) < 0 or int(species_value) < 0:
            result.append(int(block_index))
    return tuple(result)


def _multilevel_weights_for_blocks(
    layout: RHS1QICoarseBlockLayout, weighted_blocks: Sequence[tuple[int, float]]
) -> tuple[float, ...]:
    weights = [0.0 for _ in layout.block_sizes]
    for block_index, value in weighted_blocks:
        weights[int(block_index)] = float(value)
    return tuple(weights)


def _multilevel_species_block_groups(
    layout: RHS1QICoarseBlockLayout, block_indices: Sequence[int]
) -> dict[int, tuple[int, ...]]:
    groups: dict[int, list[int]] = {}
    for block_index in block_indices:
        species = int((layout.block_species or ())[int(block_index)])
        groups.setdefault(species, []).append(int(block_index))
    return {species: tuple(indices) for species, indices in groups.items()}


def _multilevel_centered_weights_for_block_indices(
    layout: RHS1QICoarseBlockLayout, block_indices: Sequence[int]
) -> tuple[tuple[int, float], ...] | None:
    if len(block_indices) <= 1:
        return None
    x_values = tuple(
        (float((layout.block_x or ())[int(index)]) for index in block_indices)
    )
    weights = _multilevel_centered_unit_weights(x_values)
    if weights is None:
        return None
    return tuple(
        (
            (int(block_index), float(weight))
            for block_index, weight in zip(block_indices, weights, strict=True)
        )
    )


def _multilevel_build_current_constraint_candidates(
    layout: RHS1QICoarseBlockLayout, *, config: RHS1QIMultilevelCoarseConfig
) -> tuple[ArrayLike, tuple[str, ...]]:
    """Build high-priority current and constraint/nullspace candidates.

    These columns target the slow RHSMode=1 channels that are easy to truncate
    when generic angular/radial candidates fill the coarse rank budget first:
    global/species bootstrap-current moments, radial variation of those current
    moments, and the reduced tail/source block.  They are structural and
    deterministic, so they remain compatible with device-side operator reuse.
    """
    dtype = config.dtype
    columns: list[ArrayLike] = []
    labels: list[str] = []
    f_blocks = _multilevel_non_tail_block_indices(layout)
    species_groups = _multilevel_species_block_groups(layout, f_blocks)
    max_degree = max(0, int(config.max_current_pitch_degree))
    if bool(config.include_current_moments) and f_blocks and (max_degree > 0):
        all_f_weights = _multilevel_weights_for_blocks(
            layout, tuple(((block_index, 1.0) for block_index in f_blocks))
        )
        radial_weights = _multilevel_centered_weights_for_block_indices(
            layout, f_blocks
        )
        for degree in range(1, max_degree + 1):
            _multilevel_append_candidate(
                columns,
                labels,
                _multilevel_block_weighted_pitch_moment(
                    layout, all_f_weights, degree=degree, dtype=dtype
                ),
                f"current:global:p{degree}",
            )
            if (
                bool(config.include_radial_current_moments)
                and radial_weights is not None
            ):
                _multilevel_append_candidate(
                    columns,
                    labels,
                    _multilevel_block_weighted_pitch_moment(
                        layout,
                        _multilevel_weights_for_blocks(layout, radial_weights),
                        degree=degree,
                        dtype=dtype,
                    ),
                    f"current:radial:p1:pitch:p{degree}",
                )
            if bool(config.include_species_current_moments):
                for species in sorted(species_groups):
                    species_blocks = species_groups[species]
                    species_weights = _multilevel_weights_for_blocks(
                        layout,
                        tuple(((block_index, 1.0) for block_index in species_blocks)),
                    )
                    _multilevel_append_candidate(
                        columns,
                        labels,
                        _multilevel_block_weighted_pitch_moment(
                            layout, species_weights, degree=degree, dtype=dtype
                        ),
                        f"current:species:{species}:p{degree}",
                    )
                    if bool(config.include_radial_current_moments):
                        species_radial_weights = (
                            _multilevel_centered_weights_for_block_indices(
                                layout, species_blocks
                            )
                        )
                        if species_radial_weights is not None:
                            _multilevel_append_candidate(
                                columns,
                                labels,
                                _multilevel_block_weighted_pitch_moment(
                                    layout,
                                    _multilevel_weights_for_blocks(
                                        layout, species_radial_weights
                                    ),
                                    degree=degree,
                                    dtype=dtype,
                                ),
                                f"current:species:{species}:radial:p1:pitch:p{degree}",
                            )
    if bool(config.include_tail_constraint_moments):
        tail_blocks = _multilevel_tail_block_indices(layout)
        if tail_blocks:
            tail_weights = _multilevel_weights_for_blocks(
                layout, tuple(((block_index, 1.0) for block_index in tail_blocks))
            )
            _multilevel_append_candidate(
                columns,
                labels,
                _multilevel_block_weighted_constant(layout, tail_weights, dtype),
                "constraint_tail:aggregate",
            )
    if not columns:
        return (_multilevel_empty_matrix(layout.total_size, dtype), ())
    return (jnp.stack(tuple(columns), axis=1), tuple(labels))


def _multilevel_group_centers(
    layout: RHS1QICoarseBlockLayout, groups: Sequence[Sequence[int]]
) -> tuple[float, ...]:
    block_x = tuple(
        (
            float(value)
            for value in layout.block_x or tuple(range(len(layout.block_sizes)))
        )
    )
    centers: list[float] = []
    for group in groups:
        group_values = tuple((block_x[int(block_index)] for block_index in group))
        centers.append(sum(group_values) / float(len(group_values)))
    return tuple(centers)


def _multilevel_append_candidate(
    columns: list[ArrayLike], labels: list[str], values: ArrayLike | None, label: str
) -> None:
    if values is not None:
        columns.append(jnp.asarray(values).reshape((-1,)))
        labels.append(str(label))


def _multilevel_is_finest_raw_aggregate_label(label: str) -> bool:
    prefix = "level:0:aggregate:"
    if not label.startswith(prefix):
        return False
    return ":" not in label[len(prefix) :]


def _multilevel_build_level_candidates(
    layout: RHS1QICoarseBlockLayout,
    groups: tuple[tuple[int, ...], ...],
    *,
    level_index: int,
    config: RHS1QIMultilevelCoarseConfig,
) -> tuple[ArrayLike, tuple[str, ...]]:
    dtype = config.dtype
    columns: list[ArrayLike] = []
    labels: list[str] = []
    prefix = f"level:{int(level_index)}"
    angular_specs = _multilevel_angular_specs(
        layout, max_angular_mode=config.max_angular_mode, dtype=dtype
    )
    pitch_specs: list[tuple[str, tuple[float, ...]]] = []
    for degree in range(1, max(0, int(config.max_pitch_degree)) + 1):
        weights = _multilevel_pitch_weights(layout, degree=degree)
        if weights is not None:
            pitch_specs.append((f"pitch:p{degree}", weights))
    if config.include_level_aggregates:
        for group_index, group in enumerate(groups):
            weights = tuple(
                (1.0 if index == group_index else 0.0 for index in range(len(groups)))
            )
            label = f"{prefix}:aggregate:{group_index}"
            _multilevel_append_candidate(
                columns,
                labels,
                _multilevel_group_weighted_constant(layout, groups, weights, dtype),
                label,
            )
            if config.include_angular:
                for angular_label, angular_values in angular_specs:
                    _multilevel_append_candidate(
                        columns,
                        labels,
                        _multilevel_group_weighted_angular(
                            layout, groups, angular_values, weights, dtype
                        ),
                        f"{label}:angular:{angular_label}",
                    )
            if config.include_pitch:
                for pitch_label, pitch_weights in pitch_specs:
                    _multilevel_append_candidate(
                        columns,
                        labels,
                        _multilevel_group_weighted_pitch(
                            layout, groups, pitch_weights, weights, dtype
                        ),
                        f"{label}:{pitch_label}",
                    )
    centers = _multilevel_group_centers(layout, groups)
    radial_weights: list[tuple[str, tuple[float, ...]]] = []
    for degree in range(1, max(0, int(config.max_radial_degree)) + 1):
        weights = _multilevel_centered_power_weights(centers, degree)
        if weights is not None:
            radial_weights.append((f"radial:p{degree}", weights))
    if config.include_radial:
        for radial_label, weights in radial_weights:
            _multilevel_append_candidate(
                columns,
                labels,
                _multilevel_group_weighted_constant(layout, groups, weights, dtype),
                f"{prefix}:{radial_label}",
            )
    if config.include_radial_angular:
        for radial_label, weights in radial_weights:
            for angular_label, angular_values in angular_specs:
                _multilevel_append_candidate(
                    columns,
                    labels,
                    _multilevel_group_weighted_angular(
                        layout, groups, angular_values, weights, dtype
                    ),
                    f"{prefix}:{radial_label}:angular:{angular_label}",
                )
    if config.include_radial_pitch:
        for radial_label, weights in radial_weights:
            for pitch_label, pitch_values in pitch_specs:
                _multilevel_append_candidate(
                    columns,
                    labels,
                    _multilevel_group_weighted_pitch(
                        layout, groups, pitch_values, weights, dtype
                    ),
                    f"{prefix}:{radial_label}:{pitch_label}",
                )
    if not columns:
        return (_multilevel_empty_matrix(layout.total_size, dtype), ())
    return (jnp.stack(tuple(columns), axis=1), tuple(labels))


def _multilevel_level_metadata(
    candidates: ArrayLike,
    labels: tuple[str, ...],
    groups: tuple[tuple[int, ...], ...],
    *,
    level_index: int,
    aggregate_size: int,
    config: RHS1QIMultilevelCoarseConfig,
) -> RHS1QIMultilevelCoarseLevelMetadata:
    basis = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=float(config.rtol),
        atol=float(config.atol),
        max_rank=max(0, int(config.max_rank)),
    )
    return RHS1QIMultilevelCoarseLevelMetadata(
        level_index=int(level_index),
        aggregate_size=int(aggregate_size),
        aggregate_count=len(groups),
        block_groups=groups,
        candidate_count=int(basis.metadata.candidate_count),
        rank=int(basis.metadata.rank),
        discarded_count=int(basis.metadata.discarded_count),
        candidate_labels=tuple((str(label) for label in labels)),
        accepted_labels=tuple((str(label) for label in basis.metadata.accepted_labels)),
    )


def build_rhs1_qi_multilevel_coarse_candidates(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIMultilevelCoarseConfig | None = None,
) -> tuple[ArrayLike, tuple[str, ...], tuple[RHS1QIMultilevelCoarseLevelMetadata, ...]]:
    """Build deterministic multilevel angular-radial coarse candidates.

    Level 0 works on individual radial blocks.  Coarser levels aggregate
    contiguous radial blocks by ``aggregate_factor`` and add aggregate constants,
    angular harmonics, radial moments, and radial-angular tensor-product modes.
    The final basis builder rank-gates the union, so redundant level content is
    safe and useful for deterministic diagnostics.
    """
    cfg = RHS1QIMultilevelCoarseConfig() if config is None else config
    columns: list[ArrayLike] = []
    labels: list[str] = []
    levels: list[RHS1QIMultilevelCoarseLevelMetadata] = []
    hierarchy = _multilevel_hierarchy_groups(len(layout.block_sizes), cfg)
    structural_candidates, structural_labels = (
        _multilevel_build_current_constraint_candidates(layout, config=cfg)
    )
    for column_index, label in enumerate(structural_labels):
        columns.append(structural_candidates[:, column_index])
        labels.append(label)
    for level_index, (aggregate_size, groups) in enumerate(hierarchy):
        level_candidates, level_labels = _multilevel_build_level_candidates(
            layout, groups, level_index=level_index, config=cfg
        )
        levels.append(
            _multilevel_level_metadata(
                level_candidates,
                level_labels,
                groups,
                level_index=level_index,
                aggregate_size=aggregate_size,
                config=cfg,
            )
        )
        for column_index, label in enumerate(level_labels):
            if (
                cfg.include_finest_blocks
                or not _multilevel_is_finest_raw_aggregate_label(label)
            ):
                columns.append(level_candidates[:, column_index])
                labels.append(label)
    if not columns:
        return (
            _multilevel_empty_matrix(layout.total_size, cfg.dtype),
            (),
            tuple(levels),
        )
    return (jnp.stack(tuple(columns), axis=1), tuple(labels), tuple(levels))


def build_rhs1_qi_multilevel_coarse_basis(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIMultilevelCoarseConfig | None = None,
) -> tuple[RHS1QICoarseBasis, tuple[RHS1QIMultilevelCoarseLevelMetadata, ...]]:
    """Build and rank-gate the multilevel angular-radial coarse basis."""
    cfg = RHS1QIMultilevelCoarseConfig() if config is None else config
    candidates, labels, levels = build_rhs1_qi_multilevel_coarse_candidates(
        layout, config=cfg
    )
    basis = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        max_rank=max(0, int(cfg.max_rank)),
    )
    return (basis, levels)


def _multilevel_normalize_nested_order(value: str) -> str:
    order = str(value).strip().lower().replace("-", "_")
    if order in {"coarse_to_fine", "coarse", "coarse_first"}:
        return "coarse_to_fine"
    if order in {"fine_to_coarse", "fine", "fine_first"}:
        return "fine_to_coarse"
    raise ValueError("nested_order must be 'coarse_to_fine' or 'fine_to_coarse'")


def _multilevel_normalize_nested_solver(value: str) -> str:
    solver = str(value).strip().lower().replace("-", "_")
    aliases = {
        "action": "action_lstsq",
        "action_ls": "action_lstsq",
        "action_lstsq": "action_lstsq",
        "least_squares": "action_lstsq",
        "lstsq": "action_lstsq",
        "staged": "action_lstsq",
        "galerkin": "galerkin",
        "projected": "galerkin",
        "qtaq": "galerkin",
        "coarse_grid": "galerkin",
    }
    if solver not in aliases:
        raise ValueError("nested_solver must be 'action_lstsq' or 'galerkin'")
    return aliases[solver]


def build_rhs1_qi_multilevel_residual_level_bases(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIMultilevelCoarseConfig | None = None,
) -> tuple[
    tuple[RHS1QICoarseBasis, ...], tuple[RHS1QIMultilevelCoarseLevelMetadata, ...]
]:
    """Build per-level bases for the nested coarse residual equation.

    Unlike the global multilevel basis, these bases are rank-gated separately by
    level. This preserves coarse-grid residual directions that can be discarded
    by a single flat rank budget, while keeping every level bounded and
    deterministic.
    """
    cfg = RHS1QIMultilevelCoarseConfig() if config is None else config
    order = _multilevel_normalize_nested_order(cfg.nested_order)
    hierarchy = tuple(
        enumerate(_multilevel_hierarchy_groups(len(layout.block_sizes), cfg))
    )
    if order == "coarse_to_fine":
        hierarchy = tuple(reversed(hierarchy))
    level_bases: list[RHS1QICoarseBasis] = []
    level_metadata: list[RHS1QIMultilevelCoarseLevelMetadata] = []
    for level_index, (aggregate_size, groups) in hierarchy:
        candidates, labels = _multilevel_build_level_candidates(
            layout, groups, level_index=int(level_index), config=cfg
        )
        max_rank = max(0, int(cfg.nested_level_max_rank))
        basis = orthonormalize_rhs1_qi_coarse_basis(
            candidates,
            labels=labels,
            rtol=float(cfg.rtol),
            atol=float(cfg.atol),
            max_rank=max_rank,
        )
        level_metadata.append(
            RHS1QIMultilevelCoarseLevelMetadata(
                level_index=int(level_index),
                aggregate_size=int(aggregate_size),
                aggregate_count=len(groups),
                block_groups=groups,
                candidate_count=int(basis.metadata.candidate_count),
                rank=int(basis.metadata.rank),
                discarded_count=int(basis.metadata.discarded_count),
                candidate_labels=tuple((str(label) for label in labels)),
                accepted_labels=tuple(
                    (str(label) for label in basis.metadata.accepted_labels)
                ),
            )
        )
        if int(basis.metadata.rank) > 0:
            level_bases.append(basis)
    return (tuple(level_bases), tuple(level_metadata))


def _multilevel_apply_operator_to_basis(
    operator: LinearOperator, basis_vectors: ArrayLike
) -> ArrayLike:
    q = jnp.asarray(basis_vectors)
    if q.ndim != 2:
        raise ValueError("basis vectors must be a 2D matrix")
    if int(q.shape[1]) == 0:
        return _multilevel_empty_matrix(int(q.shape[0]), q.dtype)
    return jnp.stack(
        tuple(
            (
                jnp.asarray(operator(q[:, index])).reshape((-1,))
                for index in range(int(q.shape[1]))
            )
        ),
        axis=1,
    )


def _multilevel_regularized_action_least_squares(
    matrix: ArrayLike, rhs: ArrayLike, *, rcond: float
) -> ArrayLike:
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


def _multilevel_regularized_galerkin_solve(
    matrix: ArrayLike, rhs: ArrayLike, *, rcond: float
) -> ArrayLike:
    """Solve a square projected residual equation with a scale-relative ridge."""
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2:
        raise ValueError("Galerkin matrix must be 2D")
    if int(a.shape[0]) != int(a.shape[1]):
        raise ValueError("Galerkin matrix must be square")
    if int(a.shape[0]) != int(rhs_vec.shape[0]):
        raise ValueError("rhs length must match Galerkin matrix rows")
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    row_sums = jnp.sum(jnp.abs(a), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=a.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=a.dtype) * scale
    eye = jnp.eye(int(a.shape[0]), dtype=a.dtype)
    return jnp.linalg.solve(a + ridge * eye, rhs_vec)


def build_rhs1_qi_multilevel_coarse_preconditioner(
    *,
    operator: LinearOperator,
    layout: RHS1QICoarseBlockLayout | None = None,
    basis: RHS1QICoarseBasis | None = None,
    level_metadata: tuple[RHS1QIMultilevelCoarseLevelMetadata, ...] = (),
    local_smoother: LinearOperator | None = None,
    config: RHS1QIMultilevelCoarseConfig | None = None,
) -> RHS1QIMultilevelCoarsePreconditioner:
    """Build a reusable local-plus-multilevel coarse preconditioner."""
    cfg = RHS1QIMultilevelCoarseConfig() if config is None else config
    nested_solver = _multilevel_normalize_nested_solver(cfg.nested_solver)
    if basis is None:
        if layout is None:
            raise ValueError("either basis or layout must be provided")
        basis, level_metadata = build_rhs1_qi_multilevel_coarse_basis(
            layout, config=cfg
        )
    q = jnp.asarray(basis.vectors)
    if q.ndim != 2:
        raise ValueError("basis vectors must be a 2D matrix")
    smoother = (
        (lambda residual: jnp.zeros_like(jnp.asarray(residual).reshape((-1,))))
        if local_smoother is None
        else local_smoother
    )
    rank = int(q.shape[1])
    if rank <= 0:
        operator_on_basis = _multilevel_empty_matrix(int(q.shape[0]), q.dtype)
        coarse_operator = jnp.zeros((0, 0), dtype=q.dtype)
    else:
        operator_on_basis = _multilevel_apply_operator_to_basis(operator, q)
        coarse_operator = jnp.conjugate(q).T @ operator_on_basis
    nested_bases: tuple[RHS1QICoarseBasis, ...] = ()
    nested_operator_on_bases: tuple[ArrayLike, ...] = ()
    nested_coarse_operators: tuple[ArrayLike, ...] = ()
    nested_level_metadata: tuple[RHS1QIMultilevelCoarseLevelMetadata, ...] = ()
    if bool(cfg.nested_residual_correction) and layout is not None:
        nested_bases, nested_level_metadata = (
            build_rhs1_qi_multilevel_residual_level_bases(layout, config=cfg)
        )
        nested_actions: list[ArrayLike] = []
        nested_projected_operators: list[ArrayLike] = []
        for nested_basis in nested_bases:
            nested_q = jnp.asarray(nested_basis.vectors)
            nested_action = _multilevel_apply_operator_to_basis(operator, nested_q)
            nested_actions.append(nested_action)
            nested_projected_operators.append(jnp.conjugate(nested_q).T @ nested_action)
        nested_operator_on_bases = tuple(nested_actions)
        nested_coarse_operators = tuple(nested_projected_operators)
    nested_level_ranks = tuple(
        (int(nested_basis.metadata.rank) for nested_basis in nested_bases)
    )
    nested_rank = sum(nested_level_ranks)
    nested_enabled = bool(cfg.nested_residual_correction) and nested_rank > 0
    nested_coarse_operator_shapes = tuple(
        (
            tuple((int(v) for v in nested_coarse_operator.shape))
            for nested_coarse_operator in nested_coarse_operators
        )
    )
    nested_coarse_operator_norms = tuple(
        (
            float(jnp.linalg.norm(nested_coarse_operator))
            for nested_coarse_operator in nested_coarse_operators
        )
    )
    if nested_enabled and nested_solver == "galerkin":
        reason = "built_with_nested_galerkin_residual_equation"
    elif nested_enabled:
        reason = "built_with_nested_residual_equation"
    elif rank > 0:
        reason = "built_with_multilevel_coarse"
    else:
        reason = "empty_basis"
    metadata = RHS1QIMultilevelCoarseMetadata(
        total_size=int(q.shape[0]),
        level_count=len(level_metadata),
        levels=tuple(level_metadata),
        candidate_count=int(basis.metadata.candidate_count),
        rank=rank,
        nested_residual_correction_enabled=nested_enabled,
        nested_level_count=len(nested_bases),
        nested_rank=int(nested_rank),
        nested_level_ranks=nested_level_ranks,
        nested_order=_multilevel_normalize_nested_order(cfg.nested_order),
        nested_solver=nested_solver,
        nested_include_global=bool(cfg.nested_include_global),
        nested_coarse_operator_shapes=nested_coarse_operator_shapes,
        nested_coarse_operator_norms=nested_coarse_operator_norms,
        discarded_count=int(basis.metadata.discarded_count),
        operator_on_basis_shape=tuple((int(v) for v in operator_on_basis.shape)),
        operator_on_basis_norm=float(jnp.linalg.norm(operator_on_basis))
        if rank > 0
        else 0.0,
        coarse_operator_shape=tuple((int(v) for v in coarse_operator.shape)),
        coarse_operator_norm=float(jnp.linalg.norm(coarse_operator))
        if rank > 0
        else 0.0,
        regularization_rcond=float(cfg.regularization_rcond),
        damping=float(cfg.damping),
        accepted_labels=tuple((str(label) for label in basis.metadata.accepted_labels)),
        candidate_labels=tuple(
            (str(label) for label in basis.metadata.candidate_labels)
        ),
        device_resident=True,
        host_callback_free=True,
        reason=reason,
    )
    return RHS1QIMultilevelCoarsePreconditioner(
        operator=operator,
        local_smoother=smoother,
        basis=basis,
        operator_on_basis=operator_on_basis,
        coarse_operator=coarse_operator,
        nested_bases=nested_bases,
        nested_operator_on_bases=nested_operator_on_bases,
        nested_coarse_operators=nested_coarse_operators,
        metadata=metadata,
    )


def probe_rhs1_qi_multilevel_coarse_correction(
    *,
    operator: LinearOperator,
    rhs: ArrayLike,
    x0: ArrayLike,
    preconditioner: RHS1QIMultilevelCoarsePreconditioner,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
) -> tuple[ArrayLike, RHS1QIMultilevelCoarseProbe]:
    """Apply one correction and accept it only if true residual decreases."""
    rhs_vec = jnp.asarray(rhs).reshape((-1,))
    x_initial = jnp.asarray(x0).reshape((-1,))
    if rhs_vec.shape != x_initial.shape:
        raise ValueError("rhs and x0 must have the same shape")
    residual_before = rhs_vec - jnp.asarray(operator(x_initial)).reshape((-1,))
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    if residual_before_norm == 0.0:
        probe = RHS1QIMultilevelCoarseProbe(
            accepted=False,
            reason="zero_residual",
            residual_before_norm=0.0,
            residual_after_norm=0.0,
            improvement_ratio=None,
            metadata=preconditioner.metadata,
        )
        return (x_initial, probe)
    if (
        int(preconditioner.metadata.rank) <= 0
        and int(preconditioner.metadata.nested_rank) <= 0
    ):
        probe = RHS1QIMultilevelCoarseProbe(
            accepted=False,
            reason="empty_basis",
            residual_before_norm=residual_before_norm,
            residual_after_norm=residual_before_norm,
            improvement_ratio=1.0,
            metadata=preconditioner.metadata,
        )
        return (x_initial, probe)
    dx = jnp.asarray(preconditioner.apply(residual_before)).reshape((-1,))
    candidate = x_initial + dx
    residual_after = rhs_vec - jnp.asarray(operator(candidate)).reshape((-1,))
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = residual_after_norm / residual_before_norm
    required_drop = max(
        float(acceptance_atol),
        residual_before_norm * max(0.0, float(min_relative_improvement)),
    )
    finite = bool(jnp.isfinite(jnp.asarray(residual_after_norm)))
    accepted = finite and residual_after_norm < residual_before_norm - required_drop
    if accepted:
        reason = "residual_reduced"
    elif not finite:
        reason = "nonfinite_candidate"
    else:
        reason = "not_reduced"
    probe = RHS1QIMultilevelCoarseProbe(
        accepted=bool(accepted),
        reason=reason,
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm if finite else residual_before_norm,
        improvement_ratio=float(improvement_ratio) if finite else None,
        metadata=preconditioner.metadata,
    )
    return (candidate if accepted else x_initial, probe)


@dataclass(frozen=True)
class RHS1QIResidualGalerkinConfig:
    """Static controls for residual-derived coarse-space construction."""

    max_stages: int = 3
    max_stage_rank: int | None = 4
    max_rank: int | None = None
    min_rank: int = 1
    rank_rtol: float = 1e-10
    rank_atol: float = 0.0
    regularization_rcond: float = 1e-12
    damping: float = 1.0
    solver: str = "action_lstsq"
    min_relative_improvement: float = 0.0
    acceptance_atol: float = 0.0
    include_global_residual: bool = False
    include_block_residuals: bool = True
    include_operator_images: bool = False
    include_operator_preimages: bool = False
    sort_blocks_by_residual_norm: bool = True


@dataclass(frozen=True)
class RHS1QIResidualGalerkinMetadata:
    """JSON-friendly setup diagnostics for the residual Galerkin state."""

    rank: int
    stage_count: int
    stage_ranks: tuple[int, ...]
    candidate_count: int
    residual_before: float
    residual_after: float
    labels: tuple[str, ...]
    solver: str
    condition_estimate: float
    accepted: bool
    reason: str
    candidate_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return plain Python containers suitable for traces or JSON."""
        return asdict(self)


@dataclass(frozen=True)
class RHS1QIResidualGalerkinState:
    """Reusable pure-JAX residual coarse correction.

    ``basis`` stores the accepted solution-space basis ``Q`` and
    ``operator_on_basis`` stores the cached action ``A @ Q``.  Applying the
    state to any residual uses only those cached arrays and a small JAX dense
    solve, so the apply path is safe to close over in ``jax.jit``.
    """

    basis: ArrayLike
    operator_on_basis: ArrayLike
    coarse_operator: ArrayLike
    regularization_rcond: float
    damping: float
    solver: str
    metadata: RHS1QIResidualGalerkinMetadata

    def solve_coefficients(self, residual: ArrayLike) -> ArrayLike:
        """Solve the cached coarse problem for ``residual``."""
        residual_vec = jnp.asarray(residual, dtype=self.basis.dtype).reshape((-1,))
        rank = int(self.metadata.rank)
        if rank <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        if self.solver == "action_lstsq":
            return _residualgalerkin_regularized_least_squares(
                self.operator_on_basis,
                residual_vec,
                rcond=float(self.regularization_rcond),
            )
        if self.solver == "galerkin":
            projected = jnp.conjugate(self.basis).T @ residual_vec
            return _residualgalerkin_regularized_least_squares(
                self.coarse_operator, projected, rcond=float(self.regularization_rcond)
            )
        raise ValueError(f"unsupported residual Galerkin solver: {self.solver!r}")

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Return the coarse correction for ``residual``."""
        residual_vec = jnp.asarray(residual, dtype=self.basis.dtype).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return jnp.zeros_like(residual_vec)
        coefficients = self.solve_coefficients(residual_vec)
        return jnp.asarray(self.damping, dtype=residual_vec.dtype) * (
            self.basis @ coefficients
        )

    def residual_after_apply(self, residual: ArrayLike) -> ArrayLike:
        """Return the residual remaining after applying the cached coarse action."""
        residual_vec = jnp.asarray(residual, dtype=self.basis.dtype).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return residual_vec
        coefficients = self.solve_coefficients(residual_vec)
        action = jnp.asarray(self.damping, dtype=residual_vec.dtype) * (
            self.operator_on_basis @ coefficients
        )
        return residual_vec - action

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return ``apply`` for preconditioner call sites."""
        return self.apply


def setup_rhs1_qi_residual_galerkin(
    operator: LinearOperator,
    residual: ArrayLike,
    *,
    block_sizes: Sequence[int] | None = None,
    config: RHS1QIResidualGalerkinConfig | None = None,
) -> RHS1QIResidualGalerkinState:
    """Build a residual-derived, fail-closed Galerkin coarse state.

    ``residual`` is the remaining true operator residual after any outer setup
    step, for example ``rhs - A @ x_smoother``.  The returned state is accepted
    only when its cached action reduces this setup residual.
    """
    cfg = config or RHS1QIResidualGalerkinConfig()
    solver = str(cfg.solver)
    if solver not in {"action_lstsq", "galerkin"}:
        raise ValueError("solver must be 'action_lstsq' or 'galerkin'")
    residual_vec = jnp.asarray(residual).reshape((-1,))
    total_size = int(residual_vec.shape[0])
    blocks = _residualgalerkin_normalize_block_sizes(block_sizes, total_size)
    residual_before = float(jnp.linalg.norm(residual_vec))
    empty_state = _residualgalerkin_empty_state(
        total_size,
        residual_vec.dtype,
        cfg=cfg,
        residual_before=residual_before,
        reason="zero_residual" if residual_before == 0.0 else "empty_basis",
        candidate_count=0,
        candidate_labels=(),
    )
    if residual_before == 0.0:
        return empty_state
    stage_residual = residual_vec
    accepted_columns: list[ArrayLike] = []
    accepted_actions: list[ArrayLike] = []
    accepted_labels: list[str] = []
    all_candidate_labels: list[str] = []
    stage_ranks: list[int] = []
    candidate_count = 0
    rejection_reason = "empty_basis"
    max_stages = max(0, int(cfg.max_stages))
    max_total_rank = None if cfg.max_rank is None else max(0, int(cfg.max_rank))
    for stage_index in range(max_stages):
        if max_total_rank is not None and len(accepted_columns) >= max_total_rank:
            break
        candidates, labels = _residualgalerkin_build_stage_candidates(
            operator, stage_residual, blocks=blocks, stage_index=stage_index, cfg=cfg
        )
        candidate_count += int(candidates.shape[1])
        all_candidate_labels.extend(labels)
        if int(candidates.shape[1]) == 0:
            rejection_reason = "empty_basis"
            break
        remaining_rank = None
        if max_total_rank is not None:
            remaining_rank = max(0, max_total_rank - len(accepted_columns))
        stage_rank_limit = _residualgalerkin_stage_rank_limit(
            cfg.max_stage_rank, remaining_rank, int(candidates.shape[1])
        )
        q_stage, labels_stage = _residualgalerkin_orthonormalize_against(
            candidates,
            labels=labels,
            existing_columns=accepted_columns,
            rtol=float(cfg.rank_rtol),
            atol=float(cfg.rank_atol),
            max_rank=stage_rank_limit,
        )
        stage_rank = int(q_stage.shape[1])
        if stage_rank <= 0:
            rejection_reason = "rank_deficient"
            break
        aq_stage = _residualgalerkin_apply_operator_to_basis(operator, q_stage)
        stage_coefficients = _residualgalerkin_solve_cached(
            q_stage,
            aq_stage,
            stage_residual,
            solver=solver,
            rcond=float(cfg.regularization_rcond),
        )
        stage_action = jnp.asarray(float(cfg.damping), dtype=stage_residual.dtype) * (
            aq_stage @ stage_coefficients
        )
        trial_residual = stage_residual - stage_action
        trial_norm = float(jnp.linalg.norm(trial_residual))
        stage_norm = float(jnp.linalg.norm(stage_residual))
        if not _residualgalerkin_is_reduced(
            before=stage_norm,
            after=trial_norm,
            min_relative_improvement=float(cfg.min_relative_improvement),
            acceptance_atol=float(cfg.acceptance_atol),
        ):
            rejection_reason = "not_reduced"
            break
        for column_index in range(stage_rank):
            accepted_columns.append(q_stage[:, column_index])
            accepted_actions.append(aq_stage[:, column_index])
        accepted_labels.extend(labels_stage)
        stage_ranks.append(stage_rank)
        stage_residual = trial_residual
    if not accepted_columns:
        return _residualgalerkin_empty_state(
            total_size,
            residual_vec.dtype,
            cfg=cfg,
            residual_before=residual_before,
            reason=rejection_reason,
            candidate_count=candidate_count,
            candidate_labels=tuple(all_candidate_labels),
            stage_ranks=tuple(stage_ranks),
        )
    q = jnp.stack(tuple(accepted_columns), axis=1)
    aq = jnp.stack(tuple(accepted_actions), axis=1)
    rank = int(q.shape[1])
    if rank < max(0, int(cfg.min_rank)):
        return _residualgalerkin_empty_state(
            total_size,
            residual_vec.dtype,
            cfg=cfg,
            residual_before=residual_before,
            reason="rank_deficient",
            candidate_count=candidate_count,
            candidate_labels=tuple(all_candidate_labels),
            stage_ranks=tuple(stage_ranks),
        )
    coarse_operator = jnp.conjugate(q).T @ aq
    coefficients = _residualgalerkin_solve_cached(
        q, aq, residual_vec, solver=solver, rcond=float(cfg.regularization_rcond)
    )
    residual_after_vec = residual_vec - jnp.asarray(
        float(cfg.damping), dtype=residual_vec.dtype
    ) * (aq @ coefficients)
    residual_after = float(jnp.linalg.norm(residual_after_vec))
    if not _residualgalerkin_is_reduced(
        before=residual_before,
        after=residual_after,
        min_relative_improvement=float(cfg.min_relative_improvement),
        acceptance_atol=float(cfg.acceptance_atol),
    ):
        return _residualgalerkin_empty_state(
            total_size,
            residual_vec.dtype,
            cfg=cfg,
            residual_before=residual_before,
            reason="not_reduced",
            candidate_count=candidate_count,
            candidate_labels=tuple(all_candidate_labels),
            stage_ranks=tuple(stage_ranks),
        )
    conditioning_matrix = aq if solver == "action_lstsq" else coarse_operator
    metadata = RHS1QIResidualGalerkinMetadata(
        rank=rank,
        stage_count=len(stage_ranks),
        stage_ranks=tuple(stage_ranks),
        candidate_count=candidate_count,
        residual_before=residual_before,
        residual_after=residual_after,
        labels=tuple(accepted_labels),
        solver=solver,
        condition_estimate=_residualgalerkin_condition_estimate(conditioning_matrix),
        accepted=True,
        reason="residual_reduced",
        candidate_labels=tuple(all_candidate_labels),
    )
    return RHS1QIResidualGalerkinState(
        basis=q,
        operator_on_basis=aq,
        coarse_operator=coarse_operator,
        regularization_rcond=float(cfg.regularization_rcond),
        damping=float(cfg.damping),
        solver=solver,
        metadata=metadata,
    )


build_rhs1_qi_residual_galerkin = setup_rhs1_qi_residual_galerkin


def _residualgalerkin_normalize_block_sizes(
    block_sizes: Sequence[int] | None, total_size: int
) -> tuple[int, ...]:
    if block_sizes is None:
        return (int(total_size),)
    blocks = tuple((int(size) for size in block_sizes))
    if not blocks:
        raise ValueError("block_sizes must contain at least one block")
    if any((size <= 0 for size in blocks)):
        raise ValueError("block_sizes must be positive")
    if sum(blocks) != int(total_size):
        raise ValueError("block_sizes must sum to the residual length")
    return blocks


def _residualgalerkin_empty_matrix(n_rows: int, dtype: Any) -> ArrayLike:
    return jnp.zeros((int(n_rows), 0), dtype=dtype)


def _residualgalerkin_empty_state(
    total_size: int,
    dtype: Any,
    *,
    cfg: RHS1QIResidualGalerkinConfig,
    residual_before: float,
    reason: str,
    candidate_count: int,
    candidate_labels: tuple[str, ...],
    stage_ranks: tuple[int, ...] = (),
) -> RHS1QIResidualGalerkinState:
    basis = _residualgalerkin_empty_matrix(total_size, dtype)
    coarse_operator = jnp.zeros((0, 0), dtype=dtype)
    metadata = RHS1QIResidualGalerkinMetadata(
        rank=0,
        stage_count=len(stage_ranks),
        stage_ranks=stage_ranks,
        candidate_count=int(candidate_count),
        residual_before=float(residual_before),
        residual_after=float(residual_before),
        labels=(),
        solver=str(cfg.solver),
        condition_estimate=float("inf"),
        accepted=False,
        reason=str(reason),
        candidate_labels=tuple(candidate_labels),
    )
    return RHS1QIResidualGalerkinState(
        basis=basis,
        operator_on_basis=basis,
        coarse_operator=coarse_operator,
        regularization_rcond=float(cfg.regularization_rcond),
        damping=float(cfg.damping),
        solver=str(cfg.solver),
        metadata=metadata,
    )


def _residualgalerkin_block_offsets(blocks: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for size in blocks:
        offsets.append(offsets[-1] + int(size))
    return tuple(offsets)


def _residualgalerkin_build_stage_candidates(
    operator: LinearOperator,
    residual: ArrayLike,
    *,
    blocks: Sequence[int],
    stage_index: int,
    cfg: RHS1QIResidualGalerkinConfig,
) -> tuple[ArrayLike, tuple[str, ...]]:
    residual_vec = jnp.asarray(residual).reshape((-1,))
    total_size = int(residual_vec.shape[0])
    columns: list[ArrayLike] = []
    labels: list[str] = []
    if bool(cfg.include_global_residual):
        columns.append(residual_vec)
        labels.append(f"stage:{stage_index}:residual")
    if bool(cfg.include_block_residuals):
        block_columns: list[tuple[float, int, ArrayLike, str]] = []
        offsets = _residualgalerkin_block_offsets(blocks)
        for block_index in range(len(blocks)):
            start = int(offsets[block_index])
            stop = int(offsets[block_index + 1])
            values = (
                jnp.zeros((total_size,), dtype=residual_vec.dtype)
                .at[start:stop]
                .set(residual_vec[start:stop])
            )
            block_norm = float(jnp.linalg.norm(values))
            block_columns.append(
                (
                    block_norm,
                    block_index,
                    values,
                    f"stage:{stage_index}:block:{block_index}:residual",
                )
            )
        if bool(cfg.sort_blocks_by_residual_norm):
            block_columns = sorted(block_columns, key=lambda item: (-item[0], item[1]))
        for _, _, values, label in block_columns:
            columns.append(values)
            labels.append(label)
    base_count = len(columns)
    if bool(cfg.include_operator_images):
        for column_index in range(base_count):
            image = _residualgalerkin_apply_operator(operator, columns[column_index])
            if int(image.shape[0]) == total_size:
                columns.append(image)
                labels.append(f"{labels[column_index]}:operator_image")
    if bool(cfg.include_operator_preimages) and (not callable(operator)):
        operator_matrix = jnp.asarray(operator)
        if operator_matrix.ndim == 2 and tuple(
            (int(v) for v in operator_matrix.shape)
        ) == (total_size, total_size):
            for column_index in range(base_count):
                preimage = _residualgalerkin_regularized_least_squares(
                    operator_matrix,
                    columns[column_index],
                    rcond=float(cfg.regularization_rcond),
                )
                columns.append(preimage)
                labels.append(f"{labels[column_index]}:operator_preimage")
    if not columns:
        return (_residualgalerkin_empty_matrix(total_size, residual_vec.dtype), ())
    return (jnp.stack(tuple(columns), axis=1), tuple(labels))


def _residualgalerkin_stage_rank_limit(
    max_stage_rank: int | None, remaining_rank: int | None, candidate_count: int
) -> int:
    limit = int(candidate_count)
    if max_stage_rank is not None:
        limit = min(limit, max(0, int(max_stage_rank)))
    if remaining_rank is not None:
        limit = min(limit, max(0, int(remaining_rank)))
    return limit


def _residualgalerkin_orthonormalize_against(
    candidates: ArrayLike,
    *,
    labels: Sequence[str],
    existing_columns: Sequence[ArrayLike],
    rtol: float,
    atol: float,
    max_rank: int,
) -> tuple[ArrayLike, tuple[str, ...]]:
    matrix = jnp.asarray(candidates)
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1, 1))
    if matrix.ndim != 2:
        raise ValueError("candidates must be a vector or matrix")
    if len(labels) != int(matrix.shape[1]):
        raise ValueError("labels must match candidate columns")
    if int(max_rank) <= 0:
        return (_residualgalerkin_empty_matrix(int(matrix.shape[0]), matrix.dtype), ())
    candidate_norms = tuple(
        (float(jnp.linalg.norm(matrix[:, idx])) for idx in range(int(matrix.shape[1])))
    )
    reference_norm = max(candidate_norms, default=0.0)
    threshold = max(float(atol), float(rtol) * reference_norm)
    q_columns: list[ArrayLike] = []
    accepted_labels: list[str] = []
    for column_index in range(int(matrix.shape[1])):
        if len(q_columns) >= int(max_rank):
            break
        vector = matrix[:, column_index]
        if float(jnp.linalg.norm(vector)) <= threshold:
            continue
        residual = vector
        for _ in range(2):
            for existing in existing_columns:
                residual = residual - existing * jnp.vdot(existing, residual)
            for q_col in q_columns:
                residual = residual - q_col * jnp.vdot(q_col, residual)
        residual_norm = float(jnp.linalg.norm(residual))
        if residual_norm <= threshold:
            continue
        q_columns.append(residual / residual_norm)
        accepted_labels.append(str(labels[column_index]))
    if not q_columns:
        return (_residualgalerkin_empty_matrix(int(matrix.shape[0]), matrix.dtype), ())
    return (jnp.stack(tuple(q_columns), axis=1), tuple(accepted_labels))


def _residualgalerkin_apply_operator(
    operator: LinearOperator, vector: ArrayLike
) -> ArrayLike:
    vec = jnp.asarray(vector).reshape((-1,))
    if callable(operator):
        return jnp.asarray(operator(vec)).reshape((-1,))
    return (jnp.asarray(operator) @ vec).reshape((-1,))


def _residualgalerkin_apply_operator_to_basis(
    operator: LinearOperator, basis: ArrayLike
) -> ArrayLike:
    q = jnp.asarray(basis)
    if q.ndim != 2:
        raise ValueError("basis must be a matrix")
    if int(q.shape[1]) == 0:
        return _residualgalerkin_empty_matrix(int(q.shape[0]), q.dtype)
    if not callable(operator):
        return jnp.asarray(operator) @ q
    columns = tuple(
        (
            _residualgalerkin_apply_operator(operator, q[:, idx])
            for idx in range(int(q.shape[1]))
        )
    )
    return jnp.stack(columns, axis=1)


def _residualgalerkin_regularized_least_squares(
    matrix: ArrayLike, rhs: ArrayLike, *, rcond: float
) -> ArrayLike:
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2:
        raise ValueError("matrix must be 2D")
    if int(rhs_vec.shape[0]) != int(a.shape[0]):
        raise ValueError("rhs length must match matrix rows")
    n_cols = int(a.shape[1])
    if n_cols == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    a_h = jnp.conjugate(a).T
    gram = a_h @ a
    projected_rhs = a_h @ rhs_vec
    row_sums = jnp.sum(jnp.abs(gram), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=row_sums.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=row_sums.dtype) * scale
    return jnp.linalg.solve(
        gram + ridge * jnp.eye(n_cols, dtype=gram.dtype), projected_rhs
    )


def _residualgalerkin_solve_cached(
    basis: ArrayLike,
    operator_on_basis: ArrayLike,
    residual: ArrayLike,
    *,
    solver: str,
    rcond: float,
) -> ArrayLike:
    if int(jnp.asarray(basis).shape[1]) == 0:
        return jnp.zeros((0,), dtype=jnp.asarray(residual).dtype)
    if solver == "action_lstsq":
        return _residualgalerkin_regularized_least_squares(
            operator_on_basis, residual, rcond=rcond
        )
    if solver == "galerkin":
        q = jnp.asarray(basis)
        aq = jnp.asarray(operator_on_basis)
        return _residualgalerkin_regularized_least_squares(
            jnp.conjugate(q).T @ aq, jnp.conjugate(q).T @ residual, rcond=rcond
        )
    raise ValueError(f"unsupported residual Galerkin solver: {solver!r}")


def _residualgalerkin_is_reduced(
    *,
    before: float,
    after: float,
    min_relative_improvement: float,
    acceptance_atol: float,
) -> bool:
    if not (math.isfinite(float(before)) and math.isfinite(float(after))):
        return False
    required_drop = max(
        float(acceptance_atol),
        float(before) * max(0.0, float(min_relative_improvement)),
    )
    return float(after) < float(before) - required_drop


def _residualgalerkin_condition_estimate(matrix: ArrayLike) -> float:
    a = jnp.asarray(matrix)
    if a.ndim != 2 or min(int(a.shape[0]), int(a.shape[1])) == 0:
        return float("inf")
    singular_values = jnp.linalg.svd(a, compute_uv=False)
    if int(singular_values.shape[0]) == 0:
        return float("inf")
    largest = float(jnp.max(singular_values))
    smallest = float(jnp.min(singular_values))
    if largest == 0.0 or smallest <= 0.0:
        return float("inf")
    return largest / smallest


VALID_QI_GALERKIN_MODES = ("additive", "multiplicative")


@dataclass(frozen=True)
class RHS1QIGalerkinProbeCandidate:
    """Measured residual for one Galerkin preconditioner candidate."""

    mode: str
    damping: float
    residual_norm: float
    improvement_ratio: float | None
    reduced: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "damping": float(self.damping),
            "residual_norm": float(self.residual_norm),
            "improvement_ratio": None
            if self.improvement_ratio is None
            else float(self.improvement_ratio),
            "reduced": bool(self.reduced),
        }


@dataclass(frozen=True)
class RHS1QIGalerkinProbeSelection:
    """Fail-closed selection result for candidate Galerkin preconditioners."""

    accepted: bool
    reason: str
    selected_index: int | None
    selected_mode: str | None
    selected_damping: float | None
    residual_before_norm: float
    residual_after_norm: float | None
    improvement_ratio: float | None
    candidates: tuple[RHS1QIGalerkinProbeCandidate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "selected_index": self.selected_index,
            "selected_mode": self.selected_mode,
            "selected_damping": None
            if self.selected_damping is None
            else float(self.selected_damping),
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": None
            if self.residual_after_norm is None
            else float(self.residual_after_norm),
            "improvement_ratio": None
            if self.improvement_ratio is None
            else float(self.improvement_ratio),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def parse_rhs1_qi_galerkin_modes(
    raw: str | None, *, default: str = "auto"
) -> tuple[str, ...]:
    """Parse a mode list for the QI Galerkin probe.

    ``auto`` tries both supported compositions.  Explicit lists such as
    ``additive,multiplicative`` are accepted so bounded GPU campaigns can use
    one environment variable instead of rerunning the whole process per mode.
    Invalid tokens are ignored; if nothing valid remains the result is
    ``("additive",)``.
    """
    text = (
        (default if raw is None or not str(raw).strip() else str(raw))
        .strip()
        .lower()
        .replace("-", "_")
    )
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens or "auto" in tokens:
        return VALID_QI_GALERKIN_MODES
    modes: list[str] = []
    for token in tokens:
        if token in VALID_QI_GALERKIN_MODES and token not in modes:
            modes.append(token)
    return tuple(modes) if modes else ("additive",)


def parse_rhs1_qi_galerkin_dampings(
    raw: str | None,
    *,
    default: float = 1.0,
    auto_defaults: Sequence[float] = (1.0, 0.5, 0.25),
) -> tuple[float, ...]:
    """Parse positive damping candidates for the QI Galerkin probe."""
    if raw is None or not str(raw).strip():
        values = (
            tuple((float(value) for value in auto_defaults))
            if float(default) == 1.0
            else (float(default),)
        )
    else:
        parsed: list[float] = []
        for token in str(raw).replace(";", ",").split(","):
            if not token.strip():
                continue
            try:
                parsed.append(float(token))
            except ValueError:
                continue
        values = tuple(parsed)
    cleaned: list[float] = []
    for value in values:
        if np.isfinite(value) and value >= 0.0 and (value not in cleaned):
            cleaned.append(float(value))
    return tuple(cleaned) if cleaned else (max(0.0, float(default)),)


def select_rhs1_qi_galerkin_probe_candidate(
    residual_before_norm: float,
    candidates: Sequence[dict[str, Any] | RHS1QIGalerkinProbeCandidate],
    *,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
) -> RHS1QIGalerkinProbeSelection:
    """Select the best candidate only if it reduces the true residual.

    The gate is intentionally conservative.  Non-finite residuals are recorded
    but cannot be selected; a candidate must beat ``residual_before_norm`` by at
    least the requested relative or absolute margin.
    """
    before = float(residual_before_norm)
    records: list[RHS1QIGalerkinProbeCandidate] = []
    for candidate in candidates:
        if isinstance(candidate, RHS1QIGalerkinProbeCandidate):
            record = candidate
        else:
            residual = float(candidate.get("residual_norm", np.inf))
            ratio = (
                residual / before if before > 0.0 and np.isfinite(residual) else None
            )
            record = RHS1QIGalerkinProbeCandidate(
                mode=str(candidate.get("mode", "additive")),
                damping=float(candidate.get("damping", 1.0)),
                residual_norm=residual,
                improvement_ratio=ratio,
                reduced=bool(np.isfinite(residual) and residual < before),
            )
        records.append(record)
    if not records:
        return RHS1QIGalerkinProbeSelection(
            accepted=False,
            reason="no_probe_candidates",
            selected_index=None,
            selected_mode=None,
            selected_damping=None,
            residual_before_norm=before,
            residual_after_norm=None,
            improvement_ratio=None,
            candidates=(),
        )
    finite = [
        (idx, record)
        for idx, record in enumerate(records)
        if np.isfinite(float(record.residual_norm))
    ]
    if not finite:
        return RHS1QIGalerkinProbeSelection(
            accepted=False,
            reason="no_finite_probe_candidates",
            selected_index=None,
            selected_mode=None,
            selected_damping=None,
            residual_before_norm=before,
            residual_after_norm=None,
            improvement_ratio=None,
            candidates=tuple(records),
        )
    best_index, best = min(finite, key=lambda item: float(item[1].residual_norm))
    required_drop = max(
        float(acceptance_atol), before * max(0.0, float(min_relative_improvement))
    )
    accepted = bool(float(best.residual_norm) < before - required_drop)
    return RHS1QIGalerkinProbeSelection(
        accepted=accepted,
        reason="probe_reduced" if accepted else "probe_not_reduced",
        selected_index=int(best_index) if accepted else None,
        selected_mode=best.mode if accepted else None,
        selected_damping=float(best.damping) if accepted else None,
        residual_before_norm=before,
        residual_after_norm=float(best.residual_norm),
        improvement_ratio=best.improvement_ratio,
        candidates=tuple(records),
    )


def _twolevel_normalize_coarse_solver(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "projected": "projected",
        "galerkin": "projected",
        "qtaq": "projected",
        "action_lstsq": "action_lstsq",
        "action_ls": "action_lstsq",
        "least_squares": "action_lstsq",
        "lstsq": "action_lstsq",
    }
    if normalized not in aliases:
        raise ValueError(
            f"coarse_solver must be one of 'projected' or 'action_lstsq' (got {value!r})"
        )
    return aliases[normalized]


@dataclass(frozen=True)
class RHS1QITwoLevelMetadata:
    """Diagnostics for a two-level QI preconditioner candidate."""

    rank: int
    coarse_operator_shape: tuple[int, int]
    coarse_operator_norm: float
    operator_on_basis_shape: tuple[int, int]
    operator_on_basis_norm: float
    regularization_rcond: float
    damping: float
    coarse_solver: str = "projected"
    composition: str = "multiplicative"

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": int(self.rank),
            "coarse_operator_shape": tuple(
                (int(v) for v in self.coarse_operator_shape)
            ),
            "coarse_operator_norm": float(self.coarse_operator_norm),
            "operator_on_basis_shape": tuple(
                (int(v) for v in self.operator_on_basis_shape)
            ),
            "operator_on_basis_norm": float(self.operator_on_basis_norm),
            "regularization_rcond": float(self.regularization_rcond),
            "damping": float(self.damping),
            "coarse_solver": self.coarse_solver,
            "composition": self.composition,
        }


@dataclass(frozen=True)
class RHS1QITwoLevelPreconditioner:
    """Device-compatible local-smoother plus coarse-correction action."""

    operator: LinearOperator
    local_smoother: LinearOperator
    basis: RHS1QICoarseBasis
    coarse_operator: ArrayLike
    operator_on_basis: ArrayLike
    metadata: RHS1QITwoLevelMetadata

    def solve_coarse(self, residual: ArrayLike) -> ArrayLike:
        """Solve the small projected problem in regularized least-squares form."""
        residual_vec = jnp.asarray(residual).reshape((-1,))
        rank = int(self.metadata.rank)
        if rank <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        if self.metadata.coarse_solver == "action_lstsq":
            return _twolevel_regularized_coarse_solve(
                self.operator_on_basis,
                residual_vec,
                rcond=float(self.metadata.regularization_rcond),
            )
        projected = jnp.conjugate(self.basis.vectors).T @ residual_vec
        return _twolevel_regularized_coarse_solve(
            self.coarse_operator,
            projected,
            rcond=float(self.metadata.regularization_rcond),
        )

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply one multiplicative two-level preconditioner action."""
        residual_vec = jnp.asarray(residual).reshape((-1,))
        local = jnp.asarray(self.local_smoother(residual_vec)).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return float(self.metadata.damping) * local
        remaining = residual_vec - jnp.asarray(self.operator(local)).reshape((-1,))
        coarse_coefficients = self.solve_coarse(remaining)
        coarse = self.basis.vectors @ coarse_coefficients
        return float(self.metadata.damping) * (local + coarse)

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for Krylov preconditioner hooks."""
        return self.apply


@dataclass(frozen=True)
class RHS1QITwoLevelProbe:
    """True-residual probe for a two-level preconditioner candidate."""

    accepted: bool
    reason: str
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float | None
    metadata: RHS1QITwoLevelMetadata

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": float(self.residual_after_norm),
            "improvement_ratio": None
            if self.improvement_ratio is None
            else float(self.improvement_ratio),
            "metadata": self.metadata.to_dict(),
        }


def _twolevel_regularized_coarse_solve(
    a: ArrayLike, rhs: ArrayLike, *, rcond: float
) -> ArrayLike:
    a_arr = jnp.asarray(a)
    rhs_arr = jnp.asarray(rhs).reshape((-1,))
    if a_arr.size == 0:
        return jnp.zeros((0,), dtype=rhs_arr.dtype)
    gram = jnp.conjugate(a_arr).T @ a_arr
    normal_rhs = jnp.conjugate(a_arr).T @ rhs_arr
    scale = jnp.maximum(jnp.linalg.norm(gram), jnp.asarray(1.0, dtype=gram.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=gram.dtype) * scale
    eye = jnp.eye(int(gram.shape[0]), dtype=gram.dtype)
    return jnp.linalg.solve(gram + ridge * eye, normal_rhs)


def _twolevel_apply_operator_to_basis(
    operator: LinearOperator, basis_vectors: ArrayLike
) -> ArrayLike:
    q = jnp.asarray(basis_vectors)
    columns = [
        jnp.asarray(operator(q[:, i])).reshape((-1,)) for i in range(int(q.shape[1]))
    ]
    if not columns:
        return jnp.zeros_like(q)
    return jnp.stack(columns, axis=1)


def build_rhs1_qi_two_level_preconditioner(
    *,
    operator: LinearOperator,
    local_smoother: LinearOperator,
    basis: RHS1QICoarseBasis,
    regularization_rcond: float = 1e-12,
    damping: float = 1.0,
    coarse_solver: str = "projected",
) -> RHS1QITwoLevelPreconditioner:
    """Build a reusable two-level QI preconditioner from a coarse basis."""
    q = jnp.asarray(basis.vectors)
    rank = int(q.shape[1]) if q.ndim == 2 else 0
    coarse_solver_norm = _twolevel_normalize_coarse_solver(coarse_solver)
    if rank <= 0:
        coarse_operator = jnp.zeros((0, 0), dtype=q.dtype)
        operator_on_basis = jnp.zeros_like(q)
    else:
        operator_on_basis = _twolevel_apply_operator_to_basis(operator, q)
        coarse_operator = jnp.conjugate(q).T @ operator_on_basis
    metadata = RHS1QITwoLevelMetadata(
        rank=rank,
        coarse_operator_shape=tuple((int(v) for v in coarse_operator.shape)),
        coarse_operator_norm=float(jnp.linalg.norm(coarse_operator))
        if rank > 0
        else 0.0,
        operator_on_basis_shape=tuple((int(v) for v in operator_on_basis.shape)),
        operator_on_basis_norm=float(jnp.linalg.norm(operator_on_basis))
        if rank > 0
        else 0.0,
        regularization_rcond=float(regularization_rcond),
        damping=float(damping),
        coarse_solver=coarse_solver_norm,
    )
    return RHS1QITwoLevelPreconditioner(
        operator=operator,
        local_smoother=local_smoother,
        basis=basis,
        coarse_operator=coarse_operator,
        operator_on_basis=operator_on_basis,
        metadata=metadata,
    )


def probe_rhs1_qi_two_level_correction(
    *,
    operator: LinearOperator,
    rhs: ArrayLike,
    x0: ArrayLike,
    preconditioner: RHS1QITwoLevelPreconditioner,
    min_relative_improvement: float = 0.0,
) -> tuple[ArrayLike, RHS1QITwoLevelProbe]:
    """Apply one correction and accept it only if the true residual improves."""
    rhs_vec = jnp.asarray(rhs).reshape((-1,))
    x_initial = jnp.asarray(x0).reshape((-1,))
    residual_before = rhs_vec - jnp.asarray(operator(x_initial)).reshape((-1,))
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    dx = jnp.asarray(preconditioner.apply(residual_before)).reshape((-1,))
    x_candidate = x_initial + dx
    residual_after = rhs_vec - jnp.asarray(operator(x_candidate)).reshape((-1,))
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = (
        residual_after_norm / residual_before_norm
        if residual_before_norm > 0.0
        else None
    )
    required = residual_before_norm * max(0.0, 1.0 - float(min_relative_improvement))
    accepted = bool(
        jnp.isfinite(jnp.asarray(residual_after_norm))
        and residual_after_norm < required
    )
    probe = RHS1QITwoLevelProbe(
        accepted=accepted,
        reason="residual_reduced" if accepted else "residual_not_reduced",
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm,
        improvement_ratio=improvement_ratio,
        metadata=preconditioner.metadata,
    )
    return (x_candidate if accepted else x_initial, probe)


def build_rhs1_xblock_two_level_preconditioner(
    *,
    op: Any,
    rhs: ArrayLike,
    matvec: LinearOperator,
    base_preconditioner: LinearOperator,
    direction_projector: LinearOperator | None = None,
    expected_size: int | None = None,
    mode: str,
    fsavg_lmax: int,
    max_extra_units: int,
    max_directions: int,
    rcond: float,
    include_rhs: bool,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[LinearOperator, dict[str, object], dict[str, int]]:
    """Wrap an x-block preconditioner with a fixed global coarse inverse."""
    from scipy.linalg import qr, solve_triangular

    mode_use = str(mode).strip().lower().replace("-", "_")
    if mode_use not in {"additive", "multiplicative"}:
        mode_use = "additive"
    max_dirs_use = max(1, int(max_directions))
    expected_size_use = (
        int(op.total_size) if expected_size is None else int(expected_size)
    )
    raw_directions = build_rhs1_xblock_global_coarse_basis(
        op=op,
        rhs=rhs,
        preconditioner=base_preconditioner,
        include_rhs=bool(include_rhs),
        fsavg_lmax=int(fsavg_lmax),
        max_extra_units=int(max_extra_units),
        max_directions=max_dirs_use,
    )
    basis_cols: list[np.ndarray] = []
    abasis_cols: list[np.ndarray] = []
    names: list[str] = []
    for name, direction in raw_directions[:max_dirs_use]:
        if direction_projector is not None:
            direction = direction_projector(direction)
        direction_np = np.asarray(jax.device_get(direction), dtype=np.float64).reshape(
            (-1,)
        )
        if direction_np.shape != (expected_size_use,) or not np.all(
            np.isfinite(direction_np)
        ):
            continue
        norm = float(np.linalg.norm(direction_np))
        if not np.isfinite(norm) or norm <= 0.0:
            continue
        direction_np = direction_np / norm
        a_direction = np.asarray(
            jax.device_get(matvec(jnp.asarray(direction_np, dtype=jnp.float64))),
            dtype=np.float64,
        ).reshape((-1,))
        if a_direction.shape != direction_np.shape or not np.all(
            np.isfinite(a_direction)
        ):
            continue
        a_norm = float(np.linalg.norm(a_direction))
        if not np.isfinite(a_norm) or a_norm <= 0.0:
            continue
        names.append(str(name))
        basis_cols.append(direction_np)
        abasis_cols.append(a_direction)
    if not basis_cols:
        raise RuntimeError(
            "two-level x-block preconditioner found no valid coarse directions"
        )
    basis = np.column_stack(basis_cols)
    abasis = np.column_stack(abasis_cols)
    q, r, piv = qr(abasis, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        raise RuntimeError("two-level x-block preconditioner coarse QR is empty")
    threshold = max(float(rcond), 0.0) * max(float(diag[0]), 1.0)
    rank = int(np.count_nonzero(diag > threshold))
    if rank <= 0:
        raise RuntimeError(
            "two-level x-block preconditioner coarse QR is rank deficient"
        )
    piv_use = np.asarray(piv[:rank], dtype=np.int32)
    q_use = np.asarray(q[:, :rank], dtype=np.float64)
    r_use = np.asarray(r[:rank, :rank], dtype=np.float64)
    basis_use = np.asarray(basis[:, piv_use], dtype=np.float64)
    names_use = tuple((names[int(i)] for i in piv_use))
    stats = {"applies": 0, "coarse_applies": 0}

    def coarse_correction(vec_np: np.ndarray) -> np.ndarray:
        rhs_small = q_use.T @ np.asarray(vec_np, dtype=np.float64).reshape((-1,))
        coeff = solve_triangular(r_use, rhs_small, lower=False, check_finite=False)
        return basis_use @ np.asarray(coeff, dtype=np.float64).reshape((-1,))

    def apply(value: ArrayLike) -> ArrayLike:
        stats["applies"] += 1
        value_np = np.asarray(jax.device_get(value), dtype=np.float64).reshape((-1,))
        base = np.asarray(
            jax.device_get(
                base_preconditioner(jnp.asarray(value_np, dtype=jnp.float64))
            ),
            dtype=np.float64,
        ).reshape((-1,))
        if mode_use == "multiplicative":
            abase = np.asarray(
                jax.device_get(matvec(jnp.asarray(base, dtype=jnp.float64))),
                dtype=np.float64,
            ).reshape((-1,))
            coarse_input = value_np - abase
        else:
            coarse_input = value_np
        correction = coarse_correction(coarse_input)
        stats["coarse_applies"] += 1
        return jnp.asarray(base + correction, dtype=jnp.float64)

    metadata: dict[str, object] = {
        "mode": mode_use,
        "basis_size": int(len(basis_cols)),
        "rank": int(rank),
        "requested_max_directions": int(max_dirs_use),
        "fsavg_lmax": int(fsavg_lmax),
        "include_rhs": bool(include_rhs),
        "rcond": float(rcond),
        "basis_names": names_use,
        "active_projected": bool(direction_projector is not None),
        "expected_size": int(expected_size_use),
    }
    if emit is not None:
        emit(
            0,
            f"solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres two-level coarse built mode={mode_use} basis={len(basis_cols)} rank={rank}",
        )
    return (apply, metadata, stats)


def build_rhs1_xblock_smoothed_global_coupling_preconditioner(
    *,
    op: Any,
    rhs: ArrayLike,
    matvec: LinearOperator,
    base_preconditioner: LinearOperator,
    direction_projector: LinearOperator | None = None,
    expected_size: int | None = None,
    mode: str,
    fsavg_lmax: int,
    angular_lmax: int,
    max_extra_units: int,
    max_directions: int,
    rcond: float,
    include_rhs: bool,
    max_setup_s: float = 0.0,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[LinearOperator, dict[str, object], dict[str, int]]:
    """Build a host-QR smoothed low-rank global-coupling preconditioner."""
    from scipy.linalg import qr, solve_triangular

    mode_use = str(mode).strip().lower().replace("-", "_")
    if mode_use not in {"additive", "multiplicative"}:
        mode_use = "additive"
    smoother = (
        os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SMOOTHER", "base")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if smoother in {"none", "raw", "identity", "load", "loads"}:
        smoother = "identity"
    elif smoother in {"preconditioner", "preconditioned"}:
        smoother = "base"
    elif smoother != "base":
        smoother = "base"
    max_dirs_use = max(1, int(max_directions))
    max_setup_s_use = max(0.0, float(max_setup_s))
    setup_deadline = (
        time.perf_counter() + max_setup_s_use if max_setup_s_use > 0.0 else None
    )
    setup_budget_reached = False
    expected_size_use = (
        int(op.total_size) if expected_size is None else int(expected_size)
    )
    raw_loads = build_rhs1_xblock_global_coupling_load_basis(
        op=op,
        rhs=rhs,
        include_rhs=bool(include_rhs),
        fsavg_lmax=int(fsavg_lmax),
        angular_lmax=int(angular_lmax),
        max_extra_units=int(max_extra_units),
        max_directions=max_dirs_use,
    )
    z_cols: list[np.ndarray] = []
    az_cols: list[np.ndarray] = []
    names: list[str] = []
    for name, load in raw_loads[:max_dirs_use]:
        if direction_projector is not None:
            load = direction_projector(load)
        load_np = np.asarray(jax.device_get(load), dtype=np.float64).reshape((-1,))
        if load_np.shape != (expected_size_use,) or not np.all(np.isfinite(load_np)):
            continue
        load_norm = float(np.linalg.norm(load_np))
        if not np.isfinite(load_norm) or load_norm <= 0.0:
            continue
        if smoother == "identity":
            smoothed = np.asarray(load_np / load_norm, dtype=np.float64)
        else:
            smoothed = np.asarray(
                jax.device_get(
                    base_preconditioner(
                        jnp.asarray(load_np / load_norm, dtype=jnp.float64)
                    )
                ),
                dtype=np.float64,
            ).reshape((-1,))
        if smoothed.shape != load_np.shape or not np.all(np.isfinite(smoothed)):
            continue
        z_norm = float(np.linalg.norm(smoothed))
        if not np.isfinite(z_norm) or z_norm <= 0.0:
            continue
        smoothed = smoothed / z_norm
        a_smoothed = np.asarray(
            jax.device_get(matvec(jnp.asarray(smoothed, dtype=jnp.float64))),
            dtype=np.float64,
        ).reshape((-1,))
        if a_smoothed.shape != smoothed.shape or not np.all(np.isfinite(a_smoothed)):
            continue
        a_norm = float(np.linalg.norm(a_smoothed))
        if not np.isfinite(a_norm) or a_norm <= 0.0:
            continue
        names.append(str(name))
        z_cols.append(smoothed)
        az_cols.append(a_smoothed)
        if setup_deadline is not None and time.perf_counter() >= setup_deadline:
            setup_budget_reached = True
            if emit is not None:
                emit(
                    1,
                    f"solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres global-coupling setup budget reached after basis={len(z_cols)}/{len(raw_loads)} max_setup_s={max_setup_s_use:.3f}; using partial basis",
                )
            break
    if not z_cols:
        raise RuntimeError(
            "global-coupling x-block preconditioner found no valid smoothed directions"
        )
    z_basis = np.column_stack(z_cols)
    az_basis = np.column_stack(az_cols)
    q, r, piv = qr(az_basis, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    if diag.size == 0:
        raise RuntimeError("global-coupling x-block preconditioner coarse QR is empty")
    threshold = max(float(rcond), 0.0) * max(float(diag[0]), 1.0)
    rank = int(np.count_nonzero(diag > threshold))
    if rank <= 0:
        raise RuntimeError(
            "global-coupling x-block preconditioner coarse QR is rank deficient"
        )
    piv_use = np.asarray(piv[:rank], dtype=np.int32)
    q_use = np.asarray(q[:, :rank], dtype=np.float64)
    r_use = np.asarray(r[:rank, :rank], dtype=np.float64)
    z_use = np.asarray(z_basis[:, piv_use], dtype=np.float64)
    names_use = tuple((names[int(i)] for i in piv_use))
    stats = {"applies": 0, "coarse_applies": 0}

    def coarse_correction(vec_np: np.ndarray) -> np.ndarray:
        rhs_small = q_use.T @ np.asarray(vec_np, dtype=np.float64).reshape((-1,))
        coeff = solve_triangular(r_use, rhs_small, lower=False, check_finite=False)
        return z_use @ np.asarray(coeff, dtype=np.float64).reshape((-1,))

    def apply(value: ArrayLike) -> ArrayLike:
        stats["applies"] += 1
        value_np = np.asarray(jax.device_get(value), dtype=np.float64).reshape((-1,))
        base = np.asarray(
            jax.device_get(
                base_preconditioner(jnp.asarray(value_np, dtype=jnp.float64))
            ),
            dtype=np.float64,
        ).reshape((-1,))
        if mode_use == "multiplicative":
            abase = np.asarray(
                jax.device_get(matvec(jnp.asarray(base, dtype=jnp.float64))),
                dtype=np.float64,
            ).reshape((-1,))
            coarse_input = value_np - abase
        else:
            coarse_input = value_np
        correction = coarse_correction(coarse_input)
        stats["coarse_applies"] += 1
        return jnp.asarray(base + correction, dtype=jnp.float64)

    metadata: dict[str, object] = {
        "mode": mode_use,
        "load_basis_size": int(len(raw_loads)),
        "basis_size": int(len(z_cols)),
        "rank": int(rank),
        "requested_max_directions": int(max_dirs_use),
        "fsavg_lmax": int(fsavg_lmax),
        "angular_lmax": int(angular_lmax),
        "include_rhs": bool(include_rhs),
        "rcond": float(rcond),
        "smoother": smoother,
        "setup_budget_s": float(max_setup_s_use),
        "setup_budget_reached": bool(setup_budget_reached),
        "basis_names": names_use,
    }
    if emit is not None:
        emit(
            0,
            f"solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres global-coupling built mode={mode_use} smoother={smoother} loads={len(raw_loads)} basis={len(z_cols)} rank={rank}",
        )
    return (apply, metadata, stats)


def build_rhs1_xblock_device_global_coupling_preconditioner(
    *,
    op: Any,
    rhs: ArrayLike,
    matvec: LinearOperator,
    base_preconditioner: LinearOperator,
    direction_projector: LinearOperator | None = None,
    expected_size: int | None = None,
    mode: str,
    fsavg_lmax: int,
    angular_lmax: int,
    max_extra_units: int,
    max_directions: int,
    rcond: float,
    include_rhs: bool,
    max_setup_s: float = 0.0,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[LinearOperator, dict[str, object], dict[str, int]]:
    """Build a device-resident global-coupling correction for Krylov paths."""
    from scipy.linalg import qr

    mode_use = str(mode).strip().lower().replace("-", "_")
    if mode_use not in {"additive", "multiplicative"}:
        mode_use = "additive"
    coarse_solver = (
        os.environ.get(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_DEVICE_SOLVER", "qr"
        )
        .strip()
        .lower()
        .replace("-", "_")
    )
    if coarse_solver in {"normal", "normal_equation", "normal_equations", "ridge"}:
        coarse_solver = "normal_equations"
    elif coarse_solver not in {"qr", "pivoted_qr"}:
        coarse_solver = "qr"
    smoother = (
        os.environ.get(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SMOOTHER", "identity"
        )
        .strip()
        .lower()
        .replace("-", "_")
    )
    if smoother in {"none", "raw", "identity", "load", "loads"}:
        smoother = "identity"
    elif smoother in {"base", "preconditioner", "preconditioned"}:
        smoother = "base"
    else:
        smoother = "identity"
    expected_size_use = (
        int(op.total_size) if expected_size is None else int(expected_size)
    )
    max_dirs_use = max(1, int(max_directions))
    max_setup_s_use = max(0.0, float(max_setup_s))
    setup_deadline = (
        time.perf_counter() + max_setup_s_use if max_setup_s_use > 0.0 else None
    )
    setup_budget_reached = False
    raw_loads = build_rhs1_xblock_global_coupling_load_basis(
        op=op,
        rhs=rhs,
        include_rhs=bool(include_rhs),
        fsavg_lmax=int(fsavg_lmax),
        angular_lmax=int(angular_lmax),
        max_extra_units=int(max_extra_units),
        max_directions=max_dirs_use,
    )
    z_cols: list[ArrayLike] = []
    az_cols: list[ArrayLike] = []
    names: list[str] = []
    for name, load in raw_loads[:max_dirs_use]:
        if direction_projector is not None:
            load = direction_projector(load)
        load_j = jnp.asarray(load, dtype=jnp.float64).reshape((-1,))
        if load_j.shape != (expected_size_use,):
            continue
        load_np = np.asarray(jax.device_get(load_j), dtype=np.float64)
        if not np.all(np.isfinite(load_np)):
            continue
        load_norm = float(np.linalg.norm(load_np))
        if not np.isfinite(load_norm) or load_norm <= 0.0:
            continue
        load_unit = jnp.asarray(load_np / load_norm, dtype=jnp.float64)
        if smoother == "identity":
            z = load_unit
        else:
            z = jnp.asarray(base_preconditioner(load_unit), dtype=jnp.float64).reshape(
                (-1,)
            )
        z_np = np.asarray(jax.device_get(z), dtype=np.float64)
        if z_np.shape != load_np.shape or not np.all(np.isfinite(z_np)):
            continue
        z_norm = float(np.linalg.norm(z_np))
        if not np.isfinite(z_norm) or z_norm <= 0.0:
            continue
        z = jnp.asarray(z_np / z_norm, dtype=jnp.float64)
        az = jnp.asarray(matvec(z), dtype=jnp.float64).reshape((-1,))
        az_np = np.asarray(jax.device_get(az), dtype=np.float64)
        if az_np.shape != z_np.shape or not np.all(np.isfinite(az_np)):
            continue
        az_norm = float(np.linalg.norm(az_np))
        if not np.isfinite(az_norm) or az_norm <= 0.0:
            continue
        names.append(str(name))
        z_cols.append(z)
        az_cols.append(az)
        if setup_deadline is not None and time.perf_counter() >= setup_deadline:
            setup_budget_reached = True
            if emit is not None:
                emit(
                    1,
                    f"solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres device global-coupling setup budget reached after basis={len(z_cols)}/{len(raw_loads)} max_setup_s={max_setup_s_use:.3f}; using partial basis",
                )
            break
    if not z_cols:
        raise RuntimeError(
            "device global-coupling x-block preconditioner found no valid smoothed directions"
        )
    z_basis_np = np.column_stack(
        [np.asarray(jax.device_get(col), dtype=np.float64) for col in z_cols]
    )
    az_basis_np = np.column_stack(
        [np.asarray(jax.device_get(col), dtype=np.float64) for col in az_cols]
    )
    if not np.all(np.isfinite(z_basis_np)) or not np.all(np.isfinite(az_basis_np)):
        raise RuntimeError(
            "device global-coupling x-block preconditioner found non-finite coarse columns"
        )
    ridge_value = 0.0
    names_use = tuple(names)
    singular_values: np.ndarray
    r_diag: np.ndarray
    if coarse_solver == "normal_equations":
        z_basis = jnp.asarray(z_basis_np, dtype=jnp.float64)
        az_basis = jnp.asarray(az_basis_np, dtype=jnp.float64)
        gram = az_basis.T @ az_basis
        diag_scale = jnp.maximum(
            jnp.max(jnp.abs(jnp.diag(gram))), jnp.asarray(1.0, dtype=jnp.float64)
        )
        ridge = jnp.asarray(max(float(rcond), 1e-14), dtype=jnp.float64) * diag_scale
        coarse_inv = jnp.linalg.inv(
            gram + ridge * jnp.eye(int(gram.shape[0]), dtype=jnp.float64)
        )
        gram_eigs = jnp.maximum(
            jnp.linalg.eigvalsh(gram), jnp.asarray(0.0, dtype=jnp.float64)
        )
        singular_values = np.asarray(
            jax.device_get(jnp.sqrt(gram_eigs[::-1])), dtype=np.float64
        )
        if singular_values.size == 0 or not np.all(np.isfinite(singular_values)):
            raise RuntimeError(
                "device global-coupling x-block preconditioner coarse spectrum is invalid"
            )
        rank_threshold = max(float(rcond), 0.0) * max(
            float(np.max(singular_values, initial=0.0)), 1.0
        )
        rank = int(np.count_nonzero(singular_values > rank_threshold))
        if rank <= 0:
            raise RuntimeError(
                "device global-coupling x-block preconditioner coarse system is rank deficient"
            )
        ridge_value = float(jax.device_get(ridge))
        r_diag = singular_values

        def coarse_coeff(coarse_input: ArrayLike) -> ArrayLike:
            return coarse_inv @ (az_basis.T @ coarse_input)
    else:
        q, r, piv = qr(az_basis_np, mode="economic", pivoting=True)
        diag = np.abs(np.diag(r))
        if diag.size == 0 or not np.all(np.isfinite(diag)):
            raise RuntimeError(
                "device global-coupling x-block preconditioner coarse QR is empty"
            )
        rank_threshold = max(float(rcond), 0.0) * max(float(diag[0]), 1.0)
        rank = int(np.count_nonzero(diag > rank_threshold))
        if rank <= 0:
            raise RuntimeError(
                "device global-coupling x-block preconditioner coarse QR is rank deficient"
            )
        piv_use = np.asarray(piv[:rank], dtype=np.int32)
        q_use = jnp.asarray(q[:, :rank], dtype=jnp.float64)
        r_use = jnp.asarray(r[:rank, :rank], dtype=jnp.float64)
        z_basis = jnp.asarray(z_basis_np[:, piv_use], dtype=jnp.float64)
        r_diag = np.asarray(diag[:rank], dtype=np.float64)
        singular_values = r_diag
        names_use = tuple((names[int(i)] for i in piv_use))

        def coarse_coeff(coarse_input: ArrayLike) -> ArrayLike:
            return jnp.linalg.solve(r_use, q_use.T @ coarse_input)

    stats = {"applies": 0, "coarse_applies": 0}

    def apply(value: ArrayLike) -> ArrayLike:
        stats["applies"] += 1
        value_j = jnp.asarray(value, dtype=jnp.float64)
        base = jnp.asarray(base_preconditioner(value_j), dtype=jnp.float64)
        if mode_use == "multiplicative":
            coarse_input = value_j - jnp.asarray(matvec(base), dtype=jnp.float64)
        else:
            coarse_input = value_j
        coeff = coarse_coeff(coarse_input)
        stats["coarse_applies"] += 1
        return base + z_basis @ coeff

    metadata: dict[str, object] = {
        "mode": mode_use,
        "load_basis_size": int(len(raw_loads)),
        "basis_size": int(z_basis.shape[1]),
        "rank": int(rank),
        "requested_max_directions": int(max_dirs_use),
        "fsavg_lmax": int(fsavg_lmax),
        "angular_lmax": int(angular_lmax),
        "include_rhs": bool(include_rhs),
        "rcond": float(rcond),
        "coarse_solver": coarse_solver,
        "smoother": smoother,
        "ridge": float(ridge_value),
        "setup_budget_s": float(max_setup_s_use),
        "setup_budget_reached": bool(setup_budget_reached),
        "basis_names": names_use,
        "device_resident": True,
        "singular_values": tuple(
            (float(v) for v in singular_values[: min(int(singular_values.size), 16)])
        ),
        "r_diag": tuple((float(v) for v in r_diag[: min(int(r_diag.size), 16)])),
    }
    if emit is not None:
        emit(
            0,
            f"solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres device global-coupling built mode={mode_use} smoother={smoother} coarse_solver={coarse_solver} loads={len(raw_loads)} basis={int(z_basis.shape[1])} rank={rank}",
        )
    return (apply, metadata, stats)


__all__ = (
    "RHS1QIBlockSchurMetadata",
    "RHS1QIBlockSchurPreconditioner",
    "RHS1QIBlockSchurProbe",
    "build_rhs1_qi_block_schur_basis",
    "build_rhs1_qi_block_schur_candidates",
    "build_rhs1_qi_block_schur_preconditioner",
    "probe_rhs1_qi_block_schur_correction",
    "RHS1QICoupledResidualEquationConfig",
    "RHS1QICoupledResidualEquationMetadata",
    "RHS1QICoupledResidualEquationState",
    "setup_rhs1_qi_coupled_residual_equation",
    "RHS1QIDeflatedPreconditioner",
    "RHS1QIDeflationMetadata",
    "RHS1QIDeflationProbe",
    "build_rhs1_qi_residual_deflated_preconditioner",
    "probe_rhs1_qi_deflated_correction",
    "probe_rhs1_qi_deflated_minres_seed",
    "RHS1QIMultilevelCoarseConfig",
    "RHS1QIMultilevelCoarseLevelMetadata",
    "RHS1QIMultilevelCoarseMetadata",
    "RHS1QIMultilevelCoarsePreconditioner",
    "RHS1QIMultilevelCoarseProbe",
    "build_rhs1_qi_multilevel_coarse_basis",
    "build_rhs1_qi_multilevel_coarse_candidates",
    "build_rhs1_qi_multilevel_coarse_preconditioner",
    "build_rhs1_qi_multilevel_residual_level_bases",
    "probe_rhs1_qi_multilevel_coarse_correction",
    "RHS1QIGalerkinProbeCandidate",
    "RHS1QIGalerkinProbeSelection",
    "VALID_QI_GALERKIN_MODES",
    "parse_rhs1_qi_galerkin_dampings",
    "parse_rhs1_qi_galerkin_modes",
    "select_rhs1_qi_galerkin_probe_candidate",
    "RHS1QITwoLevelMetadata",
    "RHS1QITwoLevelPreconditioner",
    "RHS1QITwoLevelProbe",
    "build_rhs1_xblock_device_global_coupling_preconditioner",
    "build_rhs1_xblock_smoothed_global_coupling_preconditioner",
    "build_rhs1_xblock_two_level_preconditioner",
    "build_rhs1_qi_two_level_preconditioner",
    "probe_rhs1_qi_two_level_correction",
)
