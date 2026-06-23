"""Physics problem packages that orchestrate reusable operators and solvers."""

from __future__ import annotations

from .ambipolar import (
    AmbipolarIteration,
    AmbipolarProblem,
    AmbipolarResult,
    RadialCurrentDerivativeResult,
    SfincsJaxEvaluationRecord,
    SfincsJaxRadialCurrentEvaluator,
    brent_ambipolar_root,
    finite_difference_radial_current_derivative,
    solve_ambipolar_brent,
    solve_sfincs_jax_ambipolar_brent,
    validate_fortran_v3_ambipolar_constraints,
)

__all__ = (
    "AmbipolarIteration",
    "AmbipolarProblem",
    "AmbipolarResult",
    "RadialCurrentDerivativeResult",
    "SfincsJaxEvaluationRecord",
    "SfincsJaxRadialCurrentEvaluator",
    "brent_ambipolar_root",
    "finite_difference_radial_current_derivative",
    "solve_ambipolar_brent",
    "solve_sfincs_jax_ambipolar_brent",
    "validate_fortran_v3_ambipolar_constraints",
)
