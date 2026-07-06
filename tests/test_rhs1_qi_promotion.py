from __future__ import annotations

import pytest

from sfincs_jax.validation.qi_device import QIRunEvidence, evaluate_qi_production_ladder_promotion


def test_qi_production_ladder_promotion_accepts_complete_cpu_gpu_evidence() -> None:
    runs = [
        QIRunEvidence(
            seed=seed,
            backend=backend,
            resolution=(15, 31, 60, 5),
            converged=True,
            residual_ratio=1.0e-8,
            observable_rel_diff=1.0e-10,
        )
        for seed in (0, 1)
        for backend in ("cpu", "gpu")
    ]

    result = evaluate_qi_production_ladder_promotion(
        runs,
        required_seeds=(0, 1),
        max_residual_ratio=1.0e-6,
        max_observable_rel_diff=1.0e-8,
    )

    assert result.promoted is True
    assert result.failures == ()
    assert result.present_seeds == (0, 1)
    assert result.backends == ("cpu", "gpu")


def test_qi_production_ladder_promotion_rejects_incomplete_or_host_fallback_runs() -> None:
    runs = [
        QIRunEvidence(
            seed=0,
            backend="cpu",
            resolution=(15, 31, 60, 5),
            converged=True,
            residual_ratio=1.0e-8,
        ),
        QIRunEvidence(
            seed=0,
            backend="gpu",
            resolution=(15, 31, 60, 5),
            converged=True,
            residual_ratio=1.0e-8,
            host_fallback_used=True,
        ),
        QIRunEvidence(
            seed=1,
            backend="cpu",
            resolution=(9, 19, 35, 4),
            converged=False,
            residual_ratio=1.0e-2,
            timed_out=True,
            output_written=False,
            solver_trace_written=False,
        ),
    ]

    result = evaluate_qi_production_ladder_promotion(
        runs,
        required_seeds=(0, 1),
        min_resolution=(15, 31, 60, 5),
        max_residual_ratio=1.0e-6,
    )

    assert result.promoted is False
    assert any("used host fallback" in failure for failure in result.failures)
    assert any("missing seed=1 backend=gpu" in failure for failure in result.failures)
    assert any("below" in failure for failure in result.failures)
    assert any("timed out" in failure for failure in result.failures)
    assert any("did not converge" in failure for failure in result.failures)


def test_qi_production_ladder_promotion_normalizes_mapping_evidence_and_records_warnings() -> None:
    result = evaluate_qi_production_ladder_promotion(
        [
            {
                "seed": 3,
                "backend": "GPU",
                "resolution": [15, 31, 60, 5],
                "converged": True,
                "residual_ratio": 1.0e-8,
                "observable_rel_diff": 1.0e-10,
                "host_fallback_used": True,
                "output_written": True,
                "solver_trace_written": True,
            }
        ],
        required_seeds=(3,),
        required_backends=("gpu",),
        allow_host_fallback=True,
    )

    assert result.promoted is True
    assert result.failures == ()
    assert result.warnings == ("seed=3 backend=gpu used host fallback",)
    assert result.to_dict()["backends"] == ("gpu",)


def test_qi_production_ladder_promotion_rejects_bad_shapes_and_observable_mismatch() -> None:
    with pytest.raises(ValueError, match="resolution"):
        evaluate_qi_production_ladder_promotion(
            [{"seed": 0, "backend": "cpu", "resolution": [15, 31, 60]}],
            required_seeds=(0,),
        )
    with pytest.raises(ValueError, match="min_resolution"):
        evaluate_qi_production_ladder_promotion(
            [],
            required_seeds=(0,),
            min_resolution=(15, 31, 60),
        )

    evidence = QIRunEvidence(
        seed=0,
        backend="gpu",
        resolution=(15, 31, 60, 5),
        converged=True,
        residual_ratio=1.0e-8,
        observable_rel_diff=1.0e-3,
    )
    result = evaluate_qi_production_ladder_promotion(
        [evidence.to_dict()],
        required_seeds=(0,),
        required_backends=("gpu",),
        max_observable_rel_diff=1.0e-8,
    )

    assert result.promoted is False
    assert any("observable_rel_diff" in failure for failure in result.failures)
