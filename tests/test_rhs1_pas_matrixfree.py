from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.rhs1_pas_matrixfree import (
    Rhs1PasMatrixFreeConfig,
    rhs1_pas_matrixfree_acceptance_gate,
    rhs1_pas_matrixfree_correction,
    streaming_l2_norm,
)


def test_matrixfree_correction_accepts_residual_improvement() -> None:
    diag = jnp.asarray([2.0, 4.0], dtype=jnp.float32)
    rhs = jnp.asarray([2.0, 8.0], dtype=jnp.float32)
    x0 = jnp.zeros_like(rhs)

    def matvec(x):
        return diag * x

    def exact_diagonal_correction(residual):
        return residual / diag

    result = rhs1_pas_matrixfree_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        correction=exact_diagonal_correction,
        config=Rhs1PasMatrixFreeConfig(
            max_steps=1,
            min_residual_reduction=0.5,
            block_size=1,
        ),
    )

    assert result.accepted
    assert result.accepted_steps == 1
    assert result.reason == "accepted"
    assert result.residual_norm < result.initial_residual_norm
    assert result.diagnostics["reason"] == "accepted"
    assert result.diagnostics["residual_reduction"] == pytest.approx(1.0)
    np.testing.assert_allclose(np.asarray(result.x), np.asarray([1.0, 2.0], dtype=np.float32))


def test_matrixfree_correction_rejects_nonfinite_candidate_residual() -> None:
    rhs = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    x0 = jnp.zeros_like(rhs)

    def matvec(x):
        if bool(jnp.any(x != 0.0)):
            return jnp.full_like(x, jnp.nan)
        return jnp.zeros_like(x)

    def finite_correction(residual):
        return residual

    result = rhs1_pas_matrixfree_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        correction=finite_correction,
        config=Rhs1PasMatrixFreeConfig(max_steps=1),
    )

    assert not result.accepted
    assert result.accepted_steps == 0
    assert result.reason == "nonfinite-candidate-residual"
    np.testing.assert_array_equal(np.asarray(result.x), np.asarray(x0))
    assert math.isnan(result.residual_history[-1])
    assert result.diagnostics["reason"] == "nonfinite-candidate-residual"
    assert result.diagnostics["candidate_residual_norm"] is None
    assert result.diagnostics["candidate_residual_finite"] is False


def test_matrixfree_correction_reports_update_norm_limit_reject() -> None:
    rhs = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    x0 = jnp.zeros_like(rhs)

    def matvec(x):
        return x

    def oversized_correction(residual):
        return 100.0 * residual

    result = rhs1_pas_matrixfree_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        correction=oversized_correction,
        config=Rhs1PasMatrixFreeConfig(max_update_norm_ratio=1.5),
    )

    assert not result.accepted
    assert result.reason == "update-norm-too-large"
    assert result.diagnostics["update_norm"] == pytest.approx(math.sqrt(50000.0))
    assert result.diagnostics["update_norm_limit"] == pytest.approx(1.5)
    assert result.diagnostics["max_update_norm_ratio"] == pytest.approx(1.5)
    assert result.diagnostics["matrix_free_metadata"]["candidate_matvecs"] == 0


def test_matrixfree_correction_rejects_tiny_update_without_candidate_matvec() -> None:
    rhs = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    x0 = jnp.zeros_like(rhs)
    calls = {"matvec": 0, "correction": 0}

    def matvec(x):
        calls["matvec"] += 1
        return x

    def tiny_correction(residual):
        calls["correction"] += 1
        return jnp.asarray(1.0e-8, dtype=residual.dtype) * residual

    result = rhs1_pas_matrixfree_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        correction=tiny_correction,
        config=Rhs1PasMatrixFreeConfig(
            max_steps=1,
            min_update_norm_ratio=1.0e-4,
        ),
    )

    assert not result.accepted
    assert result.reason == "update-norm-too-small"
    assert result.residual_history == pytest.approx((result.initial_residual_norm, result.initial_residual_norm))
    assert result.diagnostics["matrix_free_metadata"]["candidate_matvecs"] == 0
    assert result.diagnostics["damped_update_norm"] < result.diagnostics["min_update_norm"]
    assert calls == {"matvec": 1, "correction": 1}


def test_matrixfree_correction_candidate_size_limit_rejects_before_candidate_matvec() -> None:
    rhs = jnp.ones((5,), dtype=jnp.float32)
    x0 = jnp.zeros_like(rhs)
    calls = {"matvec": 0, "correction": 0}

    def matvec(x):
        calls["matvec"] += 1
        return x

    def correction(residual):
        calls["correction"] += 1
        return residual

    result = rhs1_pas_matrixfree_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        correction=correction,
        config=Rhs1PasMatrixFreeConfig(max_candidate_elements=4),
    )

    assert not result.accepted
    assert result.reason == "candidate-size-limit-exceeded"
    assert result.diagnostics["matrix_free_metadata"]["element_count"] == 5
    assert result.diagnostics["matrix_free_metadata"]["candidate_matvecs"] == 0
    assert calls == {"matvec": 1, "correction": 1}


def test_matrixfree_correction_rejects_zero_update_without_candidate_matvec() -> None:
    rhs = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32)
    x0 = jnp.zeros_like(rhs)
    calls = {"matvec": 0, "correction": 0}

    def matvec(x):
        calls["matvec"] += 1
        return x

    def zero_correction(residual):
        calls["correction"] += 1
        return jnp.zeros_like(residual)

    result = rhs1_pas_matrixfree_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        correction=zero_correction,
        config=Rhs1PasMatrixFreeConfig(max_steps=1, block_size=1),
    )

    assert not result.accepted
    assert result.reason == "insufficient-residual-improvement"
    assert result.residual_history == pytest.approx((result.initial_residual_norm, result.initial_residual_norm))
    assert calls == {"matvec": 1, "correction": 1}


def test_matrixfree_correction_rejects_zero_omega_without_candidate_matvec() -> None:
    rhs = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    x0 = jnp.zeros_like(rhs)
    calls = {"matvec": 0, "correction": 0}

    def matvec(x):
        calls["matvec"] += 1
        return x

    def finite_correction(residual):
        calls["correction"] += 1
        return residual

    result = rhs1_pas_matrixfree_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        correction=finite_correction,
        config=Rhs1PasMatrixFreeConfig(max_steps=1, omega=0.0),
    )

    assert not result.accepted
    assert result.reason == "insufficient-residual-improvement"
    assert calls == {"matvec": 1, "correction": 1}


def test_matrixfree_correction_preserves_shape_and_dtype() -> None:
    rhs = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    x0 = jnp.zeros_like(rhs)

    def matvec(x):
        return x

    def half_residual_correction(residual):
        return jnp.asarray(0.5, dtype=residual.dtype) * residual

    result = rhs1_pas_matrixfree_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        correction=half_residual_correction,
        config=Rhs1PasMatrixFreeConfig(
            max_steps=2,
            min_residual_reduction=0.1,
            block_size=2,
        ),
    )

    assert result.accepted
    assert result.accepted_steps == 2
    assert result.x.shape == x0.shape
    assert result.x.dtype == x0.dtype
    assert len(result.residual_history) == 3
    assert result.residual_history[-1] == pytest.approx(0.25 * result.initial_residual_norm)
    assert result.diagnostics["matrix_free_metadata"]["candidate_matvecs"] == 2
    assert result.diagnostics["matrix_free_metadata"]["estimated_live_array_count"] == 5


def test_matrixfree_acceptance_gate_documents_keep_reject_reasons() -> None:
    assert rhs1_pas_matrixfree_acceptance_gate(
        initial_residual_norm=10.0,
        candidate_residual_norm=8.0,
        min_residual_reduction=0.1,
    ) == (True, "accepted")
    assert rhs1_pas_matrixfree_acceptance_gate(
        initial_residual_norm=10.0,
        candidate_residual_norm=9.5,
        min_residual_reduction=0.1,
    ) == (False, "insufficient-residual-improvement")
    assert rhs1_pas_matrixfree_acceptance_gate(
        initial_residual_norm=10.0,
        candidate_residual_norm=float("nan"),
        min_residual_reduction=0.1,
    ) == (False, "nonfinite-candidate-residual")


def test_streaming_l2_norm_matches_dense_norm() -> None:
    value = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)

    assert streaming_l2_norm(value, block_size=2) == pytest.approx(
        streaming_l2_norm(value, block_size=None)
    )


def test_streaming_l2_norm_preserves_nonfinite_with_blocks() -> None:
    assert math.isnan(streaming_l2_norm(jnp.asarray([1.0, jnp.nan]), block_size=1))
    assert math.isinf(streaming_l2_norm(jnp.asarray([1.0, jnp.inf]), block_size=1))


def test_matrixfree_config_validation() -> None:
    with pytest.raises(ValueError):
        Rhs1PasMatrixFreeConfig(max_steps=0)
    with pytest.raises(ValueError):
        Rhs1PasMatrixFreeConfig(block_size=0)
    with pytest.raises(ValueError):
        Rhs1PasMatrixFreeConfig(max_update_norm_ratio=0.0)
    with pytest.raises(ValueError):
        Rhs1PasMatrixFreeConfig(min_update_norm_ratio=-1.0)
    with pytest.raises(ValueError):
        Rhs1PasMatrixFreeConfig(max_candidate_elements=0)
