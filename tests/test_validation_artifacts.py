from __future__ import annotations

from pathlib import Path

import numpy as np

from sfincs_jax.validation_artifacts import (
    build_publication_validation_summary,
    collisionality_grid,
    collisionality_labels,
    er_nonzero_model_spread,
    er_zero_field_spread,
    fp_pas_l11_separation,
    load_collisionality_records,
    load_er_sweep_records,
)


def _artifact_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "examples" / "publication_figures" / "artifacts"


def test_collisionality_artifact_metrics_are_literature_consistent() -> None:
    for name in ("lhd_collisionality_summary.json", "w7x_collisionality_summary.json"):
        records = load_collisionality_records(_artifact_dir() / name)
        assert collisionality_labels(records) == ["Fokker-Planck", "PAS"]
        assert len(collisionality_grid(records)) == 7

        separation = fp_pas_l11_separation(records)
        assert len(separation) == 7
        assert separation[-1]["relative_to_fp"] > separation[0]["relative_to_fp"]
        assert separation[-1]["relative_to_fp"] > 5.0


def test_er_sweep_artifact_metrics_pin_zero_field_and_finite_field_behavior() -> None:
    for name in ("er_sweep_tokamak_reference_summary.json", "er_sweep_stellarator_fast_reference_summary.json"):
        records = load_er_sweep_records(_artifact_dir() / name)
        zero_spread = er_zero_field_spread(records)
        assert all(value <= 1e-12 for value in zero_spread.values())

        jhat_spreads = er_nonzero_model_spread(records, field="fsab_jhat")
        assert jhat_spreads
        assert all(np.isfinite(value) for value in jhat_spreads.values())
        assert all(value > 0.0 for value in jhat_spreads.values())


def test_publication_validation_summary_has_research_gate_payload() -> None:
    payload = build_publication_validation_summary(artifact_dir=_artifact_dir())
    assert payload["metadata"]["kind"] == "publication_validation_dashboard"
    assert "https://doi.org/10.1063/1.4870073" in payload["metadata"]["literature"]
    assert payload["collisionality"]["lhd"]["l11_high_to_low_relative_separation_ratio"] > 10.0
    assert payload["collisionality"]["w7x"]["l11_high_to_low_relative_separation_ratio"] > 10.0
    assert payload["trajectory_sweeps"]["tokamak"]["models"] == ["dkes", "full", "partial"]
    assert payload["trajectory_sweeps"]["stellarator"]["models"] == ["dkes", "full", "partial"]
