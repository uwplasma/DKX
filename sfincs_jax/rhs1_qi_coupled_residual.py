"""Coupled residual-equation coarse solve for RHSMode=1 QI hard seeds.

The earlier true-device QI experiments built several useful coarse spaces, but
most of them were applied as a cascade: solve one coarse residual equation,
subtract its action, then solve the next one.  That is cheap and deterministic,
but it freezes early coefficients before later coarse variables are allowed to
act.  A genuine Schur or multilevel coarse correction should solve the accepted
coarse variables together, because the off-diagonal coarse couplings can be the
part that matters for hard QI seeds.

This module provides that joint solve while keeping the apply path device
compatible.  Setup concatenates rank-gated basis columns from existing coarse
families, orthonormalizes them again, probes ``A Q`` once, and accepts the
combined stage only if the measured setup residual decreases.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp
import jax

from .rhs1_qi_coarse import RHS1QICoarseBasis, orthonormalize_rhs1_qi_coarse_basis


ArrayLike = Any
LinearOperator = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class RHS1QICoupledResidualEquationConfig:
    """Static controls for the joint coarse residual equation."""

    max_rank: int | None = 96
    solver: str = "action_lstsq"
    regularization_rcond: float = 1.0e-12
    rank_rtol: float = 1.0e-10
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

        residual_vec = jnp.asarray(residual, dtype=self.basis.vectors.dtype).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        if self.solver == "galerkin":
            projected = jnp.conjugate(self.basis.vectors).T @ residual_vec
            return _regularized_square_solve(
                self.coarse_operator,
                projected,
                rcond=float(self.regularization_rcond),
            )
        return _regularized_action_least_squares(
            self.operator_on_basis,
            residual_vec,
            rcond=float(self.regularization_rcond),
        )

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Return the lifted coupled coarse correction for ``residual``."""

        residual_vec = jnp.asarray(residual, dtype=self.basis.vectors.dtype).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return jnp.zeros_like(residual_vec)
        return self.basis.vectors @ self.solve_coefficients(residual_vec)

    def residual_after_apply(self, residual: ArrayLike) -> ArrayLike:
        """Return ``residual - A @ apply(residual)`` using cached ``A Q``."""

        residual_vec = jnp.asarray(residual, dtype=self.basis.vectors.dtype).reshape((-1,))
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
    solver = _normalize_solver(cfg.solver)
    residual_vec = jnp.asarray(residual).reshape((-1,))
    total_size = int(residual_vec.shape[0])
    source_stage_ranks = tuple(int(basis.metadata.rank) for basis in bases)
    residual_before = float(jnp.linalg.norm(residual_vec))
    empty = _empty_state(
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

    candidates, labels = _collect_candidates(bases, total_size=total_size, dtype=residual_vec.dtype)
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
        return _empty_state(
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

    action = _apply_operator_to_basis(operator, basis.vectors)
    coarse_operator = jnp.conjugate(basis.vectors).T @ action
    coefficients = _solve_cached(
        basis=basis.vectors,
        action=action,
        coarse_operator=coarse_operator,
        residual=residual_vec,
        solver=solver,
        rcond=float(cfg.regularization_rcond),
    )
    residual_after_vec = residual_vec - action @ coefficients
    residual_after = float(jnp.linalg.norm(residual_after_vec))
    required_after = residual_before * (1.0 - max(0.0, float(cfg.min_relative_improvement)))
    accepted = bool(residual_after <= required_after - float(cfg.acceptance_atol))
    condition_matrix = coarse_operator if solver == "galerkin" else jnp.conjugate(action).T @ action
    condition_estimate = _condition_estimate(condition_matrix)
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
        accepted_labels=tuple(str(label) for label in basis.metadata.accepted_labels) if accepted else (),
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
    rejected_basis = _empty_basis(total_size, residual_vec.dtype, labels=labels, cfg=cfg)
    return RHS1QICoupledResidualEquationState(
        basis=rejected_basis,
        operator_on_basis=jnp.zeros((total_size, 0), dtype=residual_vec.dtype),
        coarse_operator=jnp.zeros((0, 0), dtype=residual_vec.dtype),
        metadata=metadata,
        regularization_rcond=float(cfg.regularization_rcond),
        solver=solver,
    )


def _normalize_solver(value: str) -> str:
    solver = str(value).strip().lower().replace("-", "_")
    if solver in {"action", "action_ls", "least_squares", "lstsq", "staged"}:
        return "action_lstsq"
    if solver in {"galerkin", "projected", "qtaq", "coarse_grid", "schur"}:
        return "galerkin"
    if solver == "action_lstsq":
        return "action_lstsq"
    raise ValueError("solver must be 'action_lstsq' or 'galerkin'")


def _collect_candidates(
    bases: Sequence[RHS1QICoarseBasis],
    *,
    total_size: int,
    dtype: Any,
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
        return jnp.zeros((int(total_size), 0), dtype=dtype), ()
    return jnp.stack(tuple(columns), axis=1), tuple(labels)


def _apply_operator_to_basis(operator: LinearOperator, basis_vectors: ArrayLike) -> ArrayLike:
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
        # Some sparse matvec primitives still do not expose batching rules.
        # Keep those paths correct while matrix-free operators use one batched
        # trace instead of a Python loop over every coarse column.
        return jnp.stack(tuple(apply_column(q[:, index]) for index in range(int(q.shape[1]))), axis=1)


def _solve_cached(
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
        return _regularized_square_solve(coarse_operator, projected, rcond=rcond)
    return _regularized_action_least_squares(action, residual, rcond=rcond)


def _regularized_action_least_squares(matrix: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    gram = jnp.conjugate(a).T @ a
    normal_rhs = jnp.conjugate(a).T @ rhs_vec
    return _regularized_square_solve(gram, normal_rhs, rcond=rcond)


def _regularized_square_solve(matrix: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if int(a.shape[0]) == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    row_sums = jnp.sum(jnp.abs(a), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=a.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=a.dtype) * scale
    eye = jnp.eye(int(a.shape[0]), dtype=a.dtype)
    return jnp.linalg.solve(a + ridge * eye, rhs_vec)


def _condition_estimate(matrix: ArrayLike) -> float:
    a = jnp.asarray(matrix)
    if a.ndim != 2 or int(a.shape[0]) == 0 or int(a.shape[1]) == 0:
        return float("inf")
    try:
        return float(jnp.linalg.cond(a))
    except Exception:
        return float("inf")


def _empty_basis(
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


def _empty_state(
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
    basis = _empty_basis(total_size, dtype, labels=candidate_labels, cfg=cfg)
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
        candidate_labels=tuple(str(label) for label in candidate_labels),
    )
    return RHS1QICoupledResidualEquationState(
        basis=basis,
        operator_on_basis=jnp.zeros((int(total_size), 0), dtype=dtype),
        coarse_operator=jnp.zeros((0, 0), dtype=dtype),
        metadata=metadata,
        regularization_rcond=float(cfg.regularization_rcond),
        solver=solver,
    )


__all__ = [
    "RHS1QICoupledResidualEquationConfig",
    "RHS1QICoupledResidualEquationMetadata",
    "RHS1QICoupledResidualEquationState",
    "setup_rhs1_qi_coupled_residual_equation",
]
