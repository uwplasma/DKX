from __future__ import annotations

from sfincs_jax.rhs1_qi_promotion import QIRunEvidence, evaluate_qi_production_ladder_promotion


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
