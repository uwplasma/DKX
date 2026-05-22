from __future__ import annotations

import math

import jax.numpy as jnp

from sfincs_jax.rhs1_residual import (
    l2_norm_float,
    residual_converged,
    residual_target,
    safe_ratio,
)


def test_rhs1_residual_target_matches_petsc_style_gate() -> None:
    assert residual_target(atol=1.0e-12, tol=1.0e-6, rhs_norm=3.0) == 3.0e-6
    assert residual_target(atol=1.0e-4, tol=1.0e-6, rhs_norm=3.0) == 1.0e-4


def test_rhs1_l2_norm_float_and_safe_ratio_are_host_scalars() -> None:
    norm = l2_norm_float(jnp.asarray([3.0, 4.0]))

    assert isinstance(norm, float)
    assert norm == 5.0
    assert safe_ratio(2.0, 4.0) == 0.5
    assert safe_ratio(2.0, 0.0) is None
    assert safe_ratio(math.nan, 4.0) is None
    assert safe_ratio(2.0, math.inf) is None


def test_rhs1_residual_converged_requires_finite_residual_and_target() -> None:
    assert residual_converged(1.0e-8, 1.0e-7) is True
    assert residual_converged(1.0e-6, 1.0e-7) is False
    assert residual_converged(math.nan, 1.0e-7) is False
    assert residual_converged(1.0e-8, math.inf) is False
