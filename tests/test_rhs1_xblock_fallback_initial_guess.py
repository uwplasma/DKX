import jax.numpy as jnp
import numpy as np

from sfincs_jax.v3_driver import _rhs1_xblock_fallback_initial_guess


def test_fallback_initial_guess_reuses_left_candidate_that_improves_rhs():
    original = jnp.array([0.0, 0.0])
    candidate = np.array([1.0, -2.0])

    x0, started_from_candidate, improved_rhs = _rhs1_xblock_fallback_initial_guess(
        candidate=candidate,
        original_x0=original,
        rhs_shape=(2,),
        candidate_residual_norm=0.1,
        rhs_norm=1.0,
        precondition_side="left",
    )

    assert started_from_candidate is True
    assert improved_rhs is True
    np.testing.assert_allclose(np.asarray(x0), candidate)


def test_fallback_initial_guess_rejects_candidate_that_does_not_improve_rhs():
    original = jnp.array([3.0, 4.0])

    x0, started_from_candidate, improved_rhs = _rhs1_xblock_fallback_initial_guess(
        candidate=np.array([1.0, -2.0]),
        original_x0=original,
        rhs_shape=(2,),
        candidate_residual_norm=2.0,
        rhs_norm=1.0,
        precondition_side="left",
    )

    assert started_from_candidate is False
    assert improved_rhs is False
    assert x0 is original


def test_fallback_initial_guess_rejects_right_preconditioned_candidate():
    original = jnp.array([3.0, 4.0])

    x0, started_from_candidate, improved_rhs = _rhs1_xblock_fallback_initial_guess(
        candidate=np.array([1.0, -2.0]),
        original_x0=original,
        rhs_shape=(2,),
        candidate_residual_norm=0.1,
        rhs_norm=1.0,
        precondition_side="right",
    )

    assert started_from_candidate is False
    assert improved_rhs is True
    assert x0 is original


def test_fallback_initial_guess_rejects_bad_shape_or_nonfinite_candidate():
    original = jnp.array([3.0, 4.0])

    x0_bad_shape, started_bad_shape, _ = _rhs1_xblock_fallback_initial_guess(
        candidate=np.array([1.0, -2.0, 3.0]),
        original_x0=original,
        rhs_shape=(2,),
        candidate_residual_norm=0.1,
        rhs_norm=1.0,
        precondition_side="none",
    )
    x0_nonfinite, started_nonfinite, _ = _rhs1_xblock_fallback_initial_guess(
        candidate=np.array([1.0, np.nan]),
        original_x0=original,
        rhs_shape=(2,),
        candidate_residual_norm=0.1,
        rhs_norm=1.0,
        precondition_side="none",
    )

    assert started_bad_shape is False
    assert x0_bad_shape is original
    assert started_nonfinite is False
    assert x0_nonfinite is original
