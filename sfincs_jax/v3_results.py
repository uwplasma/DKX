"""Compatibility facade for historical v3 result imports."""

from __future__ import annotations

from sfincs_jax.problems.profile_response.solver_diagnostics import V3LinearSolveResult, V3NewtonKrylovResult, v3_linear_solve_result_from_payload
from sfincs_jax.problems.transport_matrix.finalize import V3TransportMatrixSolveResult

__all__ = [
    "V3LinearSolveResult",
    "V3NewtonKrylovResult",
    "V3TransportMatrixSolveResult",
    "v3_linear_solve_result_from_payload",
]
