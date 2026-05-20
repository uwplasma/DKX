"""RHSMode=1 QI global moment closure primitive.

This module is intentionally driver-independent.  It builds a tiny
current/constraint/profile moment space from a simple block layout, probes that
space with the true operator matvec, and accepts the resulting seed correction
only when the measured setup residual decreases.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


ArrayLike = Any
LinearOperator = ArrayLike | Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class RHS1QIGlobalMomentLayout:
    """Simple block layout for low-dimensional QI moment candidates."""

    block_sizes: Sequence[int]
    block_x: Sequence[float] | None = None
    block_species: Sequence[int] | None = None

    def __post_init__(self) -> None:
        block_sizes = tuple(int(size) for size in self.block_sizes)
        if not block_sizes:
            raise ValueError("block_sizes must contain at least one block")
        if any(size <= 0 for size in block_sizes):
            raise ValueError("block_sizes must be positive")
        object.__setattr__(self, "block_sizes", block_sizes)

        n_blocks = len(block_sizes)
        if self.block_x is None:
            block_x = tuple(float(index) for index in range(n_blocks))
        else:
            block_x = tuple(float(value) for value in self.block_x)
        if len(block_x) != n_blocks:
            raise ValueError("block_x must have the same length as block_sizes")
        object.__setattr__(self, "block_x", block_x)

        if self.block_species is None:
            block_species = tuple(0 for _ in range(n_blocks))
        else:
            block_species = tuple(int(value) for value in self.block_species)
        if len(block_species) != n_blocks:
            raise ValueError("block_species must have the same length as block_sizes")
        object.__setattr__(self, "block_species", block_species)

    @property
    def total_size(self) -> int:
        """Return the full vector length represented by this layout."""

        return int(sum(self.block_sizes))

    @property
    def block_offsets(self) -> tuple[int, ...]:
        """Return block starts including the final sentinel offset."""

        offsets = [0]
        for size in self.block_sizes:
            offsets.append(offsets[-1] + int(size))
        return tuple(offsets)


@dataclass(frozen=True)
class RHS1QIGlobalMomentBasisMetadata:
    """Rank-gating diagnostics for the global moment basis."""

    total_size: int
    candidate_count: int
    rank: int
    discarded_count: int
    candidate_labels: tuple[str, ...]
    accepted_labels: tuple[str, ...]
    candidate_norms: tuple[float, ...]
    accepted_norms: tuple[float, ...]
    rank_rtol: float
    rank_atol: float


@dataclass(frozen=True)
class RHS1QIGlobalMomentBasis:
    """Orthonormal global moment basis."""

    vectors: ArrayLike
    metadata: RHS1QIGlobalMomentBasisMetadata


@dataclass(frozen=True)
class RHS1QIGlobalMomentClosureMetadata:
    """Acceptance and conditioning metadata for a global moment closure."""

    accepted: bool
    reason: str
    total_size: int
    candidate_count: int
    rank: int
    numerical_rank: int
    discarded_count: int
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float | None
    labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    solver: str
    condition_estimate: float
    min_singular_value: float
    max_singular_value: float
    singular_threshold: float
    regularization_rcond: float
    damping: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly diagnostics for solver traces."""

        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "total_size": int(self.total_size),
            "candidate_count": int(self.candidate_count),
            "rank": int(self.rank),
            "numerical_rank": int(self.numerical_rank),
            "discarded_count": int(self.discarded_count),
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": float(self.residual_after_norm),
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "labels": self.labels,
            "candidate_labels": self.candidate_labels,
            "solver": self.solver,
            "condition_estimate": float(self.condition_estimate),
            "min_singular_value": float(self.min_singular_value),
            "max_singular_value": float(self.max_singular_value),
            "singular_threshold": float(self.singular_threshold),
            "regularization_rcond": float(self.regularization_rcond),
            "damping": float(self.damping),
        }


@dataclass(frozen=True)
class RHS1QIGlobalMomentClosure:
    """Reusable JAX-compatible closure action over global QI moments."""

    basis: RHS1QIGlobalMomentBasis
    operator_on_basis: ArrayLike
    coarse_operator: ArrayLike
    metadata: RHS1QIGlobalMomentClosureMetadata

    def solve_coefficients(self, residual: ArrayLike) -> ArrayLike:
        """Solve the small moment residual equation for a residual vector."""

        residual_vec = jnp.asarray(residual).reshape((-1,))
        rank = int(self.metadata.rank)
        if (not bool(self.metadata.accepted)) or rank <= 0:
            return jnp.zeros((rank,), dtype=residual_vec.dtype)
        if self.metadata.solver == "galerkin":
            projected = jnp.conjugate(self.basis.vectors).T @ residual_vec
            return _regularized_square_solve(
                self.coarse_operator,
                projected,
                rcond=float(self.metadata.regularization_rcond),
            )
        return _regularized_action_least_squares(
            self.operator_on_basis,
            residual_vec,
            rcond=float(self.metadata.regularization_rcond),
        )

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Return the lifted moment correction for a residual vector."""

        residual_vec = jnp.asarray(residual).reshape((-1,))
        if (not bool(self.metadata.accepted)) or int(self.metadata.rank) <= 0:
            return jnp.zeros_like(residual_vec)
        coefficients = self.solve_coefficients(residual_vec)
        return float(self.metadata.damping) * (self.basis.vectors @ coefficients)

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for Krylov/preconditioner hooks."""

        return self.apply


def _empty_matrix(n_rows: int, dtype: Any) -> ArrayLike:
    return jnp.zeros((int(n_rows), 0), dtype=dtype)


def _append(
    columns: list[ArrayLike],
    labels: list[str],
    values: ArrayLike | None,
    label: str,
    limit: int,
) -> None:
    if values is None or len(columns) >= int(limit):
        return
    vector = jnp.asarray(values).reshape((-1,))
    if int(vector.shape[0]) == 0:
        return
    columns.append(vector)
    labels.append(str(label))


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


def _centered_power_weights(values: Sequence[float], power: int) -> tuple[float, ...] | None:
    linear = _centered_unit_weights(values)
    if linear is None:
        return None
    powered = tuple(value ** int(power) for value in linear)
    return _centered_unit_weights(powered)


def _weights_for_blocks(
    n_blocks: int,
    weighted_blocks: Sequence[tuple[int, float]],
) -> tuple[float, ...]:
    weights = [0.0 for _ in range(int(n_blocks))]
    for block_index, value in weighted_blocks:
        weights[int(block_index)] = float(value)
    return tuple(weights)


def _block_weighted_constant(
    layout: RHS1QIGlobalMomentLayout,
    weights: Sequence[float],
    dtype: Any,
) -> ArrayLike | None:
    if len(weights) != len(layout.block_sizes):
        raise ValueError("weights must match block_sizes")
    if not any(float(value) != 0.0 for value in weights):
        return None
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    for block_index, weight in enumerate(weights):
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(float(weight))
    return values


def _species_to_blocks(layout: RHS1QIGlobalMomentLayout) -> dict[int, list[int]]:
    species_to_blocks: dict[int, list[int]] = {}
    for block_index, species in enumerate(layout.block_species or ()):
        species_to_blocks.setdefault(int(species), []).append(block_index)
    return species_to_blocks


def _centered_index_moment(size: int, power: int, dtype: Any) -> ArrayLike | None:
    size = int(size)
    if size <= 1:
        return None
    denominator = float(max(1, size - 1))
    base = tuple((2.0 * float(index) / denominator) - 1.0 for index in range(size))
    if int(power) == 1:
        values = base
    else:
        centered = _centered_unit_weights(tuple(value ** int(power) for value in base))
        if centered is None:
            return None
        values = centered
    return jnp.asarray(values, dtype=dtype)


def _block_weighted_constraint_moment(
    layout: RHS1QIGlobalMomentLayout,
    weights: Sequence[float],
    *,
    power: int,
    dtype: Any,
) -> ArrayLike | None:
    if len(weights) != len(layout.block_sizes):
        raise ValueError("weights must match block_sizes")
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    any_supported = False
    any_nonzero = False
    for block_index, block_size in enumerate(layout.block_sizes):
        weight = float(weights[block_index])
        if weight == 0.0:
            continue
        moment = _centered_index_moment(int(block_size), int(power), dtype)
        if moment is None:
            continue
        any_supported = True
        any_nonzero = True
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(weight * moment)
    if not any_supported or not any_nonzero:
        return None
    return values


def build_rhs1_qi_global_moment_candidates(
    layout: RHS1QIGlobalMomentLayout,
    *,
    max_candidates: int = 32,
    include_current: bool = True,
    include_profile: bool = True,
    include_constraint: bool = True,
    include_cross_moments: bool = True,
    include_blocks: bool = False,
    dtype: Any = jnp.float64,
) -> tuple[ArrayLike, tuple[str, ...]]:
    """Build low-dimensional current/constraint/profile moment candidates."""

    limit = max(0, int(max_candidates))
    if limit <= 0:
        return _empty_matrix(layout.total_size, dtype), ()

    columns: list[ArrayLike] = []
    labels: list[str] = []
    n_blocks = len(layout.block_sizes)
    species_to_blocks = _species_to_blocks(layout)
    block_x = tuple(float(value) for value in (layout.block_x or tuple(range(n_blocks))))
    x_ramp = _centered_unit_weights(block_x)
    x_quad = _centered_power_weights(block_x, 2)

    def add(values: ArrayLike | None, label: str) -> None:
        _append(columns, labels, values, label, limit)

    if include_current:
        add(jnp.ones((layout.total_size,), dtype=dtype), "current:global")
        species_values = sorted(species_to_blocks)
        if len(species_values) > 1:
            reference = species_values[0]
            for species in species_values[1:]:
                weights = _weights_for_blocks(
                    n_blocks,
                    tuple((block, -1.0) for block in species_to_blocks[reference])
                    + tuple((block, 1.0) for block in species_to_blocks[species]),
                )
                add(
                    _block_weighted_constant(layout, weights, dtype),
                    f"current:species_contrast:{reference}->{species}",
                )

    profile_weights: list[tuple[str, tuple[float, ...]]] = []
    if include_profile:
        if x_ramp is not None:
            profile_weights.append(("profile:x_ramp", x_ramp))
        if x_quad is not None:
            profile_weights.append(("profile:x_quad", x_quad))
        for label, weights in profile_weights:
            add(_block_weighted_constant(layout, weights, dtype), label)

    constraint_weights: list[tuple[str, tuple[float, ...], int]] = []
    if include_constraint:
        ones = tuple(1.0 for _ in range(n_blocks))
        constraint_weights.append(("constraint:m1", ones, 1))
        constraint_weights.append(("constraint:m2", ones, 2))
        for label, weights, power in constraint_weights:
            add(
                _block_weighted_constraint_moment(layout, weights, power=power, dtype=dtype),
                label,
            )

    if include_cross_moments and include_constraint:
        if include_profile:
            for profile_label, profile in profile_weights:
                add(
                    _block_weighted_constraint_moment(layout, profile, power=1, dtype=dtype),
                    f"{profile_label}*constraint:m1",
                )
                add(
                    _block_weighted_constraint_moment(layout, profile, power=2, dtype=dtype),
                    f"{profile_label}*constraint:m2",
                )
        species_values = sorted(species_to_blocks)
        if len(species_values) > 1:
            reference = species_values[0]
            for species in species_values[1:]:
                weights = _weights_for_blocks(
                    n_blocks,
                    tuple((block, -1.0) for block in species_to_blocks[reference])
                    + tuple((block, 1.0) for block in species_to_blocks[species]),
                )
                add(
                    _block_weighted_constraint_moment(layout, weights, power=1, dtype=dtype),
                    f"current:species_contrast:{reference}->{species}*constraint:m1",
                )

    if include_blocks:
        for block_index in range(n_blocks):
            weights = _weights_for_blocks(n_blocks, ((block_index, 1.0),))
            add(_block_weighted_constant(layout, weights, dtype), f"block:{block_index}")

    if not columns:
        return _empty_matrix(layout.total_size, dtype), ()
    return jnp.stack(tuple(columns), axis=1), tuple(labels)


def orthonormalize_rhs1_qi_global_moment_basis(
    candidates: ArrayLike,
    *,
    labels: Sequence[str] | None = None,
    rtol: float = 1.0e-10,
    atol: float = 0.0,
    max_rank: int | None = None,
) -> RHS1QIGlobalMomentBasis:
    """Return a deterministic modified-Gram-Schmidt moment basis."""

    matrix = jnp.asarray(candidates)
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1, 1))
    if matrix.ndim != 2:
        raise ValueError("candidates must be a vector or a matrix")

    n_rows = int(matrix.shape[0])
    n_cols = int(matrix.shape[1])
    candidate_labels = tuple(str(label) for label in (labels or tuple(f"candidate:{i}" for i in range(n_cols))))
    if len(candidate_labels) != n_cols:
        raise ValueError("labels must match the number of candidate columns")

    candidate_norms = tuple(float(jnp.linalg.norm(matrix[:, i])) for i in range(n_cols))
    reference_norm = max(candidate_norms, default=0.0)
    threshold = max(float(atol), float(rtol) * float(reference_norm))
    rank_limit = n_cols if max_rank is None else max(0, min(int(max_rank), n_cols))

    q_columns: list[ArrayLike] = []
    accepted_labels: list[str] = []
    accepted_norms: list[float] = []
    for column_index in range(n_cols):
        if len(q_columns) >= rank_limit:
            break
        vector = matrix[:, column_index]
        norm = float(jnp.linalg.norm(vector))
        if norm <= threshold:
            continue
        residual = vector
        for _ in range(2):
            for q_col in q_columns:
                residual = residual - q_col * jnp.vdot(q_col, residual)
        residual_norm = float(jnp.linalg.norm(residual))
        if residual_norm <= threshold:
            continue
        q_columns.append(residual / residual_norm)
        accepted_labels.append(candidate_labels[column_index])
        accepted_norms.append(residual_norm)

    vectors = jnp.stack(tuple(q_columns), axis=1) if q_columns else _empty_matrix(n_rows, matrix.dtype)
    metadata = RHS1QIGlobalMomentBasisMetadata(
        total_size=n_rows,
        candidate_count=n_cols,
        rank=int(vectors.shape[1]),
        discarded_count=n_cols - int(vectors.shape[1]),
        candidate_labels=candidate_labels,
        accepted_labels=tuple(accepted_labels),
        candidate_norms=candidate_norms,
        accepted_norms=tuple(accepted_norms),
        rank_rtol=float(rtol),
        rank_atol=float(atol),
    )
    return RHS1QIGlobalMomentBasis(vectors=vectors, metadata=metadata)


def build_rhs1_qi_global_moment_basis(
    layout: RHS1QIGlobalMomentLayout,
    *,
    rtol: float = 1.0e-10,
    atol: float = 0.0,
    max_rank: int | None = 16,
    dtype: Any = jnp.float64,
    **candidate_options: Any,
) -> RHS1QIGlobalMomentBasis:
    """Build and rank-gate the global moment basis from a block layout."""

    candidates, labels = build_rhs1_qi_global_moment_candidates(
        layout,
        dtype=dtype,
        **candidate_options,
    )
    return orthonormalize_rhs1_qi_global_moment_basis(
        candidates,
        labels=labels,
        rtol=rtol,
        atol=atol,
        max_rank=max_rank,
    )


def _apply_operator(operator: LinearOperator, vector: ArrayLike) -> ArrayLike:
    vector_arr = jnp.asarray(vector).reshape((-1,))
    if callable(operator):
        return jnp.asarray(operator(vector_arr)).reshape((-1,))
    return (jnp.asarray(operator) @ vector_arr).reshape((-1,))


def _apply_operator_to_basis(operator: LinearOperator, basis_vectors: ArrayLike) -> ArrayLike:
    q = jnp.asarray(basis_vectors)
    if q.ndim != 2:
        raise ValueError("basis vectors must be a 2D matrix")
    if int(q.shape[1]) == 0:
        return _empty_matrix(int(q.shape[0]), q.dtype)
    if not callable(operator):
        return jnp.asarray(operator) @ q
    columns = tuple(_apply_operator(operator, q[:, i]) for i in range(int(q.shape[1])))
    return jnp.stack(columns, axis=1)


def _regularized_square_solve(matrix: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2 or int(a.shape[0]) != int(a.shape[1]):
        raise ValueError("matrix must be square")
    if int(a.shape[0]) != int(rhs_vec.shape[0]):
        raise ValueError("rhs length must match matrix rows")
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    row_sums = jnp.sum(jnp.abs(a), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=a.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=a.dtype) * scale
    eye = jnp.eye(int(a.shape[0]), dtype=a.dtype)
    return jnp.linalg.solve(a + ridge * eye, rhs_vec)


def _regularized_action_least_squares(matrix: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2:
        raise ValueError("matrix must be 2D")
    if int(a.shape[0]) != int(rhs_vec.shape[0]):
        raise ValueError("rhs length must match matrix rows")
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    a_h = jnp.conjugate(a).T
    gram = a_h @ a
    normal_rhs = a_h @ rhs_vec
    row_sums = jnp.sum(jnp.abs(gram), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=gram.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=gram.dtype) * scale
    eye = jnp.eye(int(gram.shape[0]), dtype=gram.dtype)
    return jnp.linalg.solve(gram + ridge * eye, normal_rhs)


def _normalize_solver(solver: str) -> str:
    normalized = str(solver).strip().lower().replace("-", "_")
    if normalized in {"galerkin", "projected", "block_schur_galerkin"}:
        return "galerkin"
    if normalized in {"action_lstsq", "action_ls", "least_squares", "minres", "residual_lstsq"}:
        return "action_lstsq"
    raise ValueError("solver must be 'galerkin' or 'action_lstsq'")


def _conditioning_metadata(
    matrix: ArrayLike,
    *,
    rcond: float,
    atol: float,
) -> tuple[int, float, float, float, float]:
    a = jnp.asarray(matrix)
    if a.ndim != 2 or int(a.shape[0]) == 0 or int(a.shape[1]) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    singular_values = jnp.linalg.svd(a, compute_uv=False)
    if int(singular_values.shape[0]) == 0:
        return 0, 0.0, 0.0, 0.0, 0.0
    max_sv = float(jnp.max(singular_values))
    min_sv = float(jnp.min(singular_values))
    threshold = max(float(atol), max_sv * max(0.0, float(rcond)))
    numerical_rank = int(jnp.sum(singular_values > threshold))
    condition = 0.0 if max_sv <= 0.0 else max_sv / max(min_sv, threshold, 1.0e-300)
    return numerical_rank, condition, min_sv, max_sv, threshold


def _zero_closure(
    *,
    basis: RHS1QIGlobalMomentBasis,
    operator_on_basis: ArrayLike,
    coarse_operator: ArrayLike,
    solver: str,
    reason: str,
    residual_before_norm: float,
    regularization_rcond: float,
    damping: float,
    numerical_rank: int = 0,
    condition_estimate: float = 0.0,
    min_singular_value: float = 0.0,
    max_singular_value: float = 0.0,
    singular_threshold: float = 0.0,
) -> RHS1QIGlobalMomentClosure:
    metadata = RHS1QIGlobalMomentClosureMetadata(
        accepted=False,
        reason=reason,
        total_size=int(basis.metadata.total_size),
        candidate_count=int(basis.metadata.candidate_count),
        rank=int(basis.metadata.rank),
        numerical_rank=int(numerical_rank),
        discarded_count=int(basis.metadata.discarded_count),
        residual_before_norm=float(residual_before_norm),
        residual_after_norm=float(residual_before_norm),
        improvement_ratio=1.0 if residual_before_norm > 0.0 else None,
        labels=tuple(basis.metadata.accepted_labels),
        candidate_labels=tuple(basis.metadata.candidate_labels),
        solver=solver,
        condition_estimate=float(condition_estimate),
        min_singular_value=float(min_singular_value),
        max_singular_value=float(max_singular_value),
        singular_threshold=float(singular_threshold),
        regularization_rcond=float(regularization_rcond),
        damping=float(damping),
    )
    return RHS1QIGlobalMomentClosure(
        basis=basis,
        operator_on_basis=operator_on_basis,
        coarse_operator=coarse_operator,
        metadata=metadata,
    )


def build_rhs1_qi_global_moment_closure(
    *,
    operator: LinearOperator,
    rhs: ArrayLike,
    x0: ArrayLike | None = None,
    layout: RHS1QIGlobalMomentLayout | None = None,
    candidates: ArrayLike | None = None,
    labels: Sequence[str] | None = None,
    solver: str = "action_lstsq",
    max_candidates: int = 32,
    max_rank: int | None = 16,
    basis_rtol: float = 1.0e-10,
    basis_atol: float = 0.0,
    min_rank: int = 1,
    require_independent_candidates: bool = False,
    regularization_rcond: float = 1.0e-12,
    conditioning_atol: float = 0.0,
    damping: float = 1.0,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
    **candidate_options: Any,
) -> tuple[ArrayLike, RHS1QIGlobalMomentClosure]:
    """Build, probe, and fail-close a global moment closure correction.

    The setup residual is ``r = rhs - A x0``.  The closure forms ``A Q`` with the
    supplied operator matvec and solves either the Galerkin equation
    ``Q* A Q c = Q* r`` or the action least-squares equation
    ``min ||A Q c - r||``.  The returned solution is updated only when the true
    residual norm strictly decreases by the requested acceptance margin.
    """

    solver_use = _normalize_solver(solver)
    rhs_vec = jnp.asarray(rhs).reshape((-1,))
    x_initial = jnp.zeros_like(rhs_vec) if x0 is None else jnp.asarray(x0, dtype=rhs_vec.dtype).reshape((-1,))
    if int(x_initial.shape[0]) != int(rhs_vec.shape[0]):
        raise ValueError("x0 and rhs must have the same length")

    if candidates is None:
        if layout is None:
            raise ValueError("either layout or candidates must be provided")
        candidates, labels = build_rhs1_qi_global_moment_candidates(
            layout,
            max_candidates=max_candidates,
            dtype=rhs_vec.dtype,
            **candidate_options,
        )
    basis = orthonormalize_rhs1_qi_global_moment_basis(
        candidates,
        labels=labels,
        rtol=float(basis_rtol),
        atol=float(basis_atol),
        max_rank=max_rank,
    )
    q = jnp.asarray(basis.vectors, dtype=rhs_vec.dtype)
    if int(q.shape[0]) != int(rhs_vec.shape[0]):
        raise ValueError("basis length must match rhs length")

    rank = int(basis.metadata.rank)
    operator_on_basis = _apply_operator_to_basis(operator, q) if rank > 0 else _empty_matrix(int(rhs_vec.shape[0]), rhs_vec.dtype)
    coarse_operator = (
        jnp.conjugate(q).T @ operator_on_basis
        if rank > 0
        else jnp.zeros((0, 0), dtype=rhs_vec.dtype)
    )
    conditioning_matrix = coarse_operator if solver_use == "galerkin" else operator_on_basis
    numerical_rank, condition, min_sv, max_sv, threshold = _conditioning_metadata(
        conditioning_matrix,
        rcond=float(regularization_rcond),
        atol=float(conditioning_atol),
    )

    residual_before = rhs_vec - _apply_operator(operator, x_initial)
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    rank_limit = int(basis.metadata.candidate_count) if max_rank is None else min(int(basis.metadata.candidate_count), int(max_rank))
    candidate_rank_deficient = bool(int(basis.metadata.rank) < rank_limit)

    if residual_before_norm == 0.0:
        closure = _zero_closure(
            basis=basis,
            operator_on_basis=operator_on_basis,
            coarse_operator=coarse_operator,
            solver=solver_use,
            reason="zero_residual",
            residual_before_norm=0.0,
            regularization_rcond=float(regularization_rcond),
            damping=float(damping),
            numerical_rank=numerical_rank,
            condition_estimate=condition,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            singular_threshold=threshold,
        )
        return x_initial, closure

    if rank <= 0:
        closure = _zero_closure(
            basis=basis,
            operator_on_basis=operator_on_basis,
            coarse_operator=coarse_operator,
            solver=solver_use,
            reason="empty_basis",
            residual_before_norm=residual_before_norm,
            regularization_rcond=float(regularization_rcond),
            damping=float(damping),
        )
        return x_initial, closure

    if rank < max(1, int(min_rank)):
        closure = _zero_closure(
            basis=basis,
            operator_on_basis=operator_on_basis,
            coarse_operator=coarse_operator,
            solver=solver_use,
            reason="rank_below_min",
            residual_before_norm=residual_before_norm,
            regularization_rcond=float(regularization_rcond),
            damping=float(damping),
            numerical_rank=numerical_rank,
            condition_estimate=condition,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            singular_threshold=threshold,
        )
        return x_initial, closure

    if bool(require_independent_candidates) and candidate_rank_deficient:
        closure = _zero_closure(
            basis=basis,
            operator_on_basis=operator_on_basis,
            coarse_operator=coarse_operator,
            solver=solver_use,
            reason="candidate_rank_deficient",
            residual_before_norm=residual_before_norm,
            regularization_rcond=float(regularization_rcond),
            damping=float(damping),
            numerical_rank=numerical_rank,
            condition_estimate=condition,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            singular_threshold=threshold,
        )
        return x_initial, closure

    if int(numerical_rank) < rank:
        closure = _zero_closure(
            basis=basis,
            operator_on_basis=operator_on_basis,
            coarse_operator=coarse_operator,
            solver=solver_use,
            reason="coarse_operator_rank_deficient",
            residual_before_norm=residual_before_norm,
            regularization_rcond=float(regularization_rcond),
            damping=float(damping),
            numerical_rank=numerical_rank,
            condition_estimate=condition,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            singular_threshold=threshold,
        )
        return x_initial, closure

    if solver_use == "galerkin":
        projected = jnp.conjugate(q).T @ residual_before
        coefficients = _regularized_square_solve(
            coarse_operator,
            projected,
            rcond=float(regularization_rcond),
        )
    else:
        coefficients = _regularized_action_least_squares(
            operator_on_basis,
            residual_before,
            rcond=float(regularization_rcond),
        )
    correction = float(damping) * (q @ coefficients)
    candidate_solution = x_initial + correction
    residual_after = rhs_vec - _apply_operator(operator, candidate_solution)
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = residual_after_norm / residual_before_norm
    finite = bool(jnp.isfinite(jnp.asarray(residual_after_norm)))
    required_drop = max(float(acceptance_atol), residual_before_norm * max(0.0, float(min_relative_improvement)))
    accepted = bool(finite and residual_after_norm < residual_before_norm - required_drop)
    if accepted:
        reason = "residual_reduced"
    elif not finite:
        reason = "nonfinite_candidate"
    else:
        reason = "not_reduced"

    metadata = RHS1QIGlobalMomentClosureMetadata(
        accepted=accepted,
        reason=reason,
        total_size=int(basis.metadata.total_size),
        candidate_count=int(basis.metadata.candidate_count),
        rank=rank,
        numerical_rank=int(numerical_rank),
        discarded_count=int(basis.metadata.discarded_count),
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm if finite else residual_before_norm,
        improvement_ratio=float(improvement_ratio) if finite else None,
        labels=tuple(basis.metadata.accepted_labels),
        candidate_labels=tuple(basis.metadata.candidate_labels),
        solver=solver_use,
        condition_estimate=float(condition),
        min_singular_value=float(min_sv),
        max_singular_value=float(max_sv),
        singular_threshold=float(threshold),
        regularization_rcond=float(regularization_rcond),
        damping=float(damping),
    )
    closure = RHS1QIGlobalMomentClosure(
        basis=basis,
        operator_on_basis=operator_on_basis,
        coarse_operator=coarse_operator,
        metadata=metadata,
    )
    return candidate_solution if accepted else x_initial, closure


__all__ = [
    "RHS1QIGlobalMomentBasis",
    "RHS1QIGlobalMomentBasisMetadata",
    "RHS1QIGlobalMomentClosure",
    "RHS1QIGlobalMomentClosureMetadata",
    "RHS1QIGlobalMomentLayout",
    "build_rhs1_qi_global_moment_basis",
    "build_rhs1_qi_global_moment_candidates",
    "build_rhs1_qi_global_moment_closure",
    "orthonormalize_rhs1_qi_global_moment_basis",
]
