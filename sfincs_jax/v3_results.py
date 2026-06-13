"""Typed result objects returned by v3-compatible solve paths."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import tree_util as jtu
import numpy as np

from .solver import GMRESSolveResult
from .v3_system import V3FullSystemOperator


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3LinearSolveResult:
    """Result of a single matrix-free solve for the v3 full-system operator."""

    op: V3FullSystemOperator
    rhs: jnp.ndarray
    gmres: GMRESSolveResult
    metadata: dict[str, object] | None = None

    def tree_flatten(self):
        children = (self.op, self.rhs, self.gmres)
        aux = self.metadata
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        op, rhs, gmres_result = children
        return cls(op=op, rhs=rhs, gmres=gmres_result, metadata=aux)

    @property
    def x(self) -> jnp.ndarray:
        return self.gmres.x

    @property
    def residual_norm(self) -> jnp.ndarray:
        return self.gmres.residual_norm


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3NewtonKrylovResult:
    """Result of an experimental Newton-Krylov solve for the v3 residual."""

    op: V3FullSystemOperator
    x: jnp.ndarray
    residual_norm: jnp.ndarray
    n_newton: int
    last_linear_residual_norm: jnp.ndarray

    def tree_flatten(self):
        children = (self.op, self.x, self.residual_norm, self.last_linear_residual_norm)
        aux = int(self.n_newton)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        op, x, residual_norm, last_linear_residual_norm = children
        return cls(
            op=op,
            x=x,
            residual_norm=residual_norm,
            n_newton=int(aux),
            last_linear_residual_norm=last_linear_residual_norm,
        )


@dataclass(frozen=True)
class V3TransportMatrixSolveResult:
    """Result of assembling a RHSMode=2/3 transport matrix by looping ``whichRHS`` solves."""

    op0: V3FullSystemOperator
    transport_matrix: jnp.ndarray
    state_vectors_by_rhs: dict[int, jnp.ndarray]
    residual_norms_by_rhs: dict[int, jnp.ndarray]
    fsab_flow: jnp.ndarray
    particle_flux_vm_psi_hat: jnp.ndarray
    heat_flux_vm_psi_hat: jnp.ndarray
    elapsed_time_s: jnp.ndarray
    transport_output_fields: dict[str, np.ndarray] | None = None
    rhs_norms_by_rhs: dict[int, jnp.ndarray] | None = None
    active_size: int | None = None
    use_active_dof_mode: bool | None = None
    solver_kinds_by_rhs: dict[int, str] | None = None
    solve_methods_by_rhs: dict[int, str] | None = None
    preconditioner_kind: str | None = None
    strong_preconditioner_kind: str | None = None


__all__ = [
    "V3LinearSolveResult",
    "V3NewtonKrylovResult",
    "V3TransportMatrixSolveResult",
]
