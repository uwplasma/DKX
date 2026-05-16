from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.rhs1_pas_matrixfree import (
    Rhs1PasMatrixFreeConfig,
    plan_pas_runtime_chunks,
    rhs1_pas_matrixfree_acceptance_gate,
    rhs1_pas_matrixfree_correction,
    rhs1_pas_matrixfree_preflight_gate,
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
    metadata = result.diagnostics["matrix_free_metadata"]
    assert metadata["preflight_kind"] == "rhs1_pas_matrixfree_candidate"
    assert metadata["safe"] is True
    assert metadata["reason"] == "within-candidate-memory-limit"
    assert metadata["candidate_byte_budget_configured"] is False
    assert metadata["candidate_byte_budget_margin"] is None
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
    assert result.diagnostics["matrix_free_metadata"]["safe"] is False
    assert calls == {"matvec": 0, "correction": 0}


def test_matrixfree_correction_candidate_byte_limit_rejects_before_correction() -> None:
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
        config=Rhs1PasMatrixFreeConfig(max_candidate_bytes=99),
    )

    metadata = result.diagnostics["matrix_free_metadata"]
    assert not result.accepted
    assert result.reason == "candidate-memory-limit-exceeded"
    assert metadata["array_bytes"] == 20
    assert metadata["estimated_live_array_count"] == 5
    assert metadata["estimated_live_array_bytes"] == 100
    assert metadata["max_candidate_bytes"] == 99
    assert metadata["candidate_byte_budget_configured"] is True
    assert metadata["candidate_byte_budget_margin"] == -1
    assert metadata["preflight_kind"] == "rhs1_pas_matrixfree_candidate"
    assert metadata["candidate_matvecs"] == 0
    assert calls == {"matvec": 0, "correction": 0}


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
    assert streaming_l2_norm(value, max_chunk_bytes=8) == pytest.approx(
        streaming_l2_norm(value, block_size=None)
    )


def test_streaming_l2_norm_preserves_nonfinite_with_blocks() -> None:
    assert math.isnan(streaming_l2_norm(jnp.asarray([1.0, jnp.nan]), block_size=1))
    assert math.isinf(streaming_l2_norm(jnp.asarray([1.0, jnp.inf]), block_size=1))


def test_streaming_l2_norm_fails_closed_when_chunk_budget_cannot_hold_one_element() -> None:
    with pytest.raises(MemoryError):
        streaming_l2_norm(jnp.asarray([1.0], dtype=jnp.float32), max_chunk_bytes=7)


def test_pas_runtime_chunk_plan_is_monotone_and_bounded_by_byte_budget() -> None:
    value = jnp.zeros((10,), dtype=jnp.float32)

    tight = plan_pas_runtime_chunks(value, max_reduction_bytes=8)
    medium = plan_pas_runtime_chunks(value, max_reduction_bytes=16)
    loose = plan_pas_runtime_chunks(value, max_reduction_bytes=80)

    assert tight.safe and medium.safe and loose.safe
    assert tight.block_size == 1
    assert medium.block_size == 2
    assert loose.block_size == 10
    assert tight.block_size <= medium.block_size <= loose.block_size
    for plan in (tight, medium, loose):
        assert plan.estimated_reduction_bytes <= plan.max_reduction_bytes


def test_pas_runtime_chunk_plan_respects_requested_block_cap() -> None:
    value = jnp.zeros((10,), dtype=jnp.float32)

    plan = plan_pas_runtime_chunks(
        value,
        requested_block_size=3,
        max_reduction_bytes=80,
        max_live_bytes=200,
        live_arrays=5,
    )

    assert plan.safe
    assert plan.block_size == 3
    assert plan.estimated_live_array_bytes == 200
    assert plan.live_byte_margin == 0


def test_matrixfree_reduction_budget_rejects_before_matvec_or_correction() -> None:
    rhs = jnp.ones((2,), dtype=jnp.float32)
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
        x0=jnp.zeros_like(rhs),
        correction=correction,
        config=Rhs1PasMatrixFreeConfig(max_reduction_bytes=7),
    )

    metadata = result.diagnostics["matrix_free_metadata"]
    assert not result.accepted
    assert result.reason == "reduction-memory-limit-exceeded"
    assert metadata["candidate_matvecs"] == 0
    assert metadata["reduction_chunk_plan"]["safe"] is False
    assert calls == {"matvec": 0, "correction": 0}


def test_matrixfree_auto_chunk_plan_preserves_function_level_result() -> None:
    diag = jnp.asarray([2.0, 4.0, 8.0], dtype=jnp.float32)
    rhs = jnp.asarray([2.0, 8.0, 24.0], dtype=jnp.float32)
    x0 = jnp.zeros_like(rhs)

    def matvec(x):
        return diag * x

    def exact_diagonal_correction(residual):
        return residual / diag

    dense = rhs1_pas_matrixfree_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        correction=exact_diagonal_correction,
        config=Rhs1PasMatrixFreeConfig(max_steps=1, min_residual_reduction=0.5),
    )
    chunked = rhs1_pas_matrixfree_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        correction=exact_diagonal_correction,
        config=Rhs1PasMatrixFreeConfig(
            max_steps=1,
            min_residual_reduction=0.5,
            max_candidate_bytes=128,
        ),
    )

    assert dense.accepted and chunked.accepted
    assert chunked.diagnostics["matrix_free_metadata"]["planned_norm_block_size"] == 3
    np.testing.assert_allclose(np.asarray(chunked.x), np.asarray(dense.x))
    assert chunked.residual_norm == pytest.approx(dense.residual_norm)


def test_matrixfree_preflight_gate_reports_safe_and_reject_metadata() -> None:
    x = jnp.zeros((4,), dtype=jnp.float32)

    safe, reason, metadata = rhs1_pas_matrixfree_preflight_gate(
        x,
        config=Rhs1PasMatrixFreeConfig(block_size=2, max_candidate_bytes=64),
        live_arrays=4,
    )

    assert safe is True
    assert reason == "within-candidate-memory-limit"
    assert metadata["estimated_live_array_bytes"] == 64
    assert metadata["block_size"] == 2
    assert metadata["candidate_byte_budget_configured"] is True
    assert metadata["candidate_byte_budget_margin"] == 0
    assert metadata["preflight_kind"] == "rhs1_pas_matrixfree_candidate"
    assert metadata["planned_norm_block_size"] == 2
    assert metadata["reduction_chunk_plan"]["estimated_reduction_bytes"] == 16

    safe, reason, metadata = rhs1_pas_matrixfree_preflight_gate(
        x,
        config=Rhs1PasMatrixFreeConfig(max_candidate_elements=3),
    )

    assert safe is False
    assert reason == "candidate-size-limit-exceeded"
    assert metadata["element_count"] == 4
    assert metadata["max_candidate_elements"] == 3
    assert metadata["candidate_element_budget_configured"] is True
    assert metadata["candidate_byte_budget_configured"] is False


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
    with pytest.raises(ValueError):
        Rhs1PasMatrixFreeConfig(max_candidate_bytes=0)
    with pytest.raises(ValueError):
        Rhs1PasMatrixFreeConfig(max_reduction_bytes=0)
