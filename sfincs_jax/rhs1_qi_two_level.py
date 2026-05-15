"""Two-level RHSMode=1 QI preconditioner building blocks.

The hard QI GPU lane needs a preconditioner that improves the true residual on
device before a long Krylov solve is launched.  This module provides a small,
JAX-compatible primitive for that strategy without changing production defaults:

``local smoother`` + ``coarse Schur/Galerkin correction``.

The action is multiplicative:

``M^{-1} r = S^{-1} r + Q A_c^{-1} Q^T (r - A S^{-1} r)``,

where ``S^{-1}`` is a cheap local block smoother and ``Q`` is a rank-gated QI
coarse basis.  The helper records enough metadata for the production driver to
fail closed when a residual probe does not improve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from .rhs1_qi_coarse import RHS1QICoarseBasis

ArrayLike = Any
LinearOperator = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class RHS1QITwoLevelMetadata:
    """Diagnostics for a two-level QI preconditioner candidate."""

    rank: int
    coarse_operator_shape: tuple[int, int]
    coarse_operator_norm: float
    regularization_rcond: float
    damping: float
    composition: str = "multiplicative"

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": int(self.rank),
            "coarse_operator_shape": tuple(int(v) for v in self.coarse_operator_shape),
            "coarse_operator_norm": float(self.coarse_operator_norm),
            "regularization_rcond": float(self.regularization_rcond),
            "damping": float(self.damping),
            "composition": self.composition,
        }


@dataclass(frozen=True)
class RHS1QITwoLevelPreconditioner:
    """Device-compatible local-smoother plus coarse-correction action."""

    operator: LinearOperator
    local_smoother: LinearOperator
    basis: RHS1QICoarseBasis
    coarse_operator: ArrayLike
    metadata: RHS1QITwoLevelMetadata

    def solve_coarse(self, residual: ArrayLike) -> ArrayLike:
        """Solve the small projected problem in regularized least-squares form."""

        residual_vec = jnp.asarray(residual).reshape((-1,))
        rank = int(self.metadata.rank)
        if rank <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        projected = jnp.conjugate(self.basis.vectors).T @ residual_vec
        return _regularized_coarse_solve(
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
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "metadata": self.metadata.to_dict(),
        }


def _regularized_coarse_solve(a: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
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


def _apply_operator_to_basis(operator: LinearOperator, basis_vectors: ArrayLike) -> ArrayLike:
    q = jnp.asarray(basis_vectors)
    columns = [jnp.asarray(operator(q[:, i])).reshape((-1,)) for i in range(int(q.shape[1]))]
    if not columns:
        return jnp.zeros_like(q)
    return jnp.stack(columns, axis=1)


def build_rhs1_qi_two_level_preconditioner(
    *,
    operator: LinearOperator,
    local_smoother: LinearOperator,
    basis: RHS1QICoarseBasis,
    regularization_rcond: float = 1.0e-12,
    damping: float = 1.0,
) -> RHS1QITwoLevelPreconditioner:
    """Build a reusable two-level QI preconditioner from a coarse basis."""

    q = jnp.asarray(basis.vectors)
    rank = int(q.shape[1]) if q.ndim == 2 else 0
    if rank <= 0:
        coarse_operator = jnp.zeros((0, 0), dtype=q.dtype)
    else:
        aq = _apply_operator_to_basis(operator, q)
        coarse_operator = jnp.conjugate(q).T @ aq
    metadata = RHS1QITwoLevelMetadata(
        rank=rank,
        coarse_operator_shape=tuple(int(v) for v in coarse_operator.shape),
        coarse_operator_norm=float(jnp.linalg.norm(coarse_operator)) if rank > 0 else 0.0,
        regularization_rcond=float(regularization_rcond),
        damping=float(damping),
    )
    return RHS1QITwoLevelPreconditioner(
        operator=operator,
        local_smoother=local_smoother,
        basis=basis,
        coarse_operator=coarse_operator,
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
        residual_after_norm / residual_before_norm if residual_before_norm > 0.0 else None
    )
    required = residual_before_norm * max(0.0, 1.0 - float(min_relative_improvement))
    accepted = bool(jnp.isfinite(jnp.asarray(residual_after_norm)) and residual_after_norm < required)
    probe = RHS1QITwoLevelProbe(
        accepted=accepted,
        reason="residual_reduced" if accepted else "residual_not_reduced",
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm,
        improvement_ratio=improvement_ratio,
        metadata=preconditioner.metadata,
    )
    return x_candidate if accepted else x_initial, probe


__all__ = [
    "RHS1QITwoLevelMetadata",
    "RHS1QITwoLevelPreconditioner",
    "RHS1QITwoLevelProbe",
    "build_rhs1_qi_two_level_preconditioner",
    "probe_rhs1_qi_two_level_correction",
]
