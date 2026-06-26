from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from sfincs_jax.problems.profile_phi1_newton import advance_phi1_newton_iterate


def test_phi1_line_search_basic_takes_scaled_full_step() -> None:
    x = jnp.array([1.0, -1.0])
    s = jnp.array([2.0, 3.0])
    out = advance_phi1_newton_iterate(
        x=x,
        step_direction=s,
        residual_norm0=10.0,
        residual_fn=lambda y: y,
        accepted=[],
        mode="basic",
        step_scale=0.5,
        factor=None,
        c1=1e-4,
        maxiter=8,
    )
    np.testing.assert_allclose(np.asarray(out), np.array([2.0, 0.5]))


def test_phi1_line_search_best_chooses_lowest_finite_trial() -> None:
    x = jnp.array([0.0])
    s = jnp.array([1.0])

    def residual_fn(y):
        # Minimum at y=1.5, which is one of the fixed "best" candidates.
        return jnp.array([y[0] - 1.5])

    out = advance_phi1_newton_iterate(
        x=x,
        step_direction=s,
        residual_norm0=10.0,
        residual_fn=residual_fn,
        accepted=[],
        mode="best",
        step_scale=1.0,
        factor=None,
        c1=1e-4,
        maxiter=8,
    )
    np.testing.assert_allclose(np.asarray(out), np.array([1.5]))


def test_phi1_line_search_backtracks_and_accepts() -> None:
    x = jnp.array([0.0])
    s = jnp.array([2.0])

    def residual_fn(y):
        return jnp.array([y[0]])

    out = advance_phi1_newton_iterate(
        x=x,
        step_direction=s,
        residual_norm0=1.0,
        residual_fn=residual_fn,
        accepted=[],
        mode="petsc",
        step_scale=1.0,
        factor=0.5,
        c1=1e-4,
        maxiter=8,
    )
    # First tries at 2.0 and 1.0 fail; 0.5 satisfies the factor test.
    np.testing.assert_allclose(np.asarray(out), np.array([0.5]))


def test_phi1_line_search_falls_back_to_last_accepted_when_all_trials_nonfinite() -> None:
    x = jnp.array([0.0])
    s = jnp.array([1.0])
    accepted = [jnp.array([-3.0])]

    out = advance_phi1_newton_iterate(
        x=x,
        step_direction=s,
        residual_norm0=1.0,
        residual_fn=lambda y: jnp.array([jnp.nan]),
        accepted=accepted,
        mode="petsc",
        step_scale=1.0,
        factor=None,
        c1=1e-4,
        maxiter=4,
    )
    np.testing.assert_allclose(np.asarray(out), np.array([-3.0]))
