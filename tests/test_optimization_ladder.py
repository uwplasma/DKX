from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dkx.workflows.optimization import (
    dense_matrix_gib,
    estimate_rhs1_active_size,
    evaluate_promotion_ladder,
)


_REPO = Path(__file__).resolve().parents[1]


def _promotion_payload(er: float, *, gate_status: str = "pass") -> dict[str, object]:
    return {
        "workflow": "dkx_optimization_high_fidelity_promotion",
        "gate_status": gate_status,
        "selected_root": {
            "er": er,
            "root_type": "electron",
            "bracket": [0.25, 0.5],
            "slope": 1.0e-8,
        },
        "bootstrap_objective": 2.0e-3,
        "flux_objective": {"total": 1.0e-15},
        "failures": [] if gate_status == "pass" else ["synthetic failure"],
    }


def _write(path: Path, payload: dict[str, object]) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_rhs1_active_size_and_dense_memory_estimates_match_finite_beta_deck() -> None:
    active = estimate_rhs1_active_size(ntheta=7, nzeta=7, nxi=5, nx=4, n_species=2)

    assert active == 1964
    assert dense_matrix_gib(active) > 0.02


def test_evaluate_promotion_ladder_defers_until_production_floor(tmp_path: Path) -> None:
    config = {
        "tiers": [
            {
                "name": "low",
                "resolution": {"Ntheta": 7, "Nzeta": 7, "Nxi": 5, "NL": 4, "Nx": 4},
                "promotions": {
                    "cpu": _write(tmp_path / "low_cpu.json", _promotion_payload(0.4136)),
                    "gpu": _write(tmp_path / "low_gpu.json", _promotion_payload(0.4136000000002)),
                },
            },
            {
                "name": "mid",
                "resolution": {"Ntheta": 9, "Nzeta": 9, "Nxi": 7, "NL": 4, "Nx": 4},
                "promotions": {
                    "cpu": _write(tmp_path / "mid_cpu.json", _promotion_payload(0.4006)),
                    "fortran_v3": _write(tmp_path / "mid_fortran.json", _promotion_payload(0.4006002)),
                },
            },
        ]
    }

    summary = evaluate_promotion_ladder(config, backend_root_atol=1.0e-6, root_drift_atol=2.0e-2)

    assert summary["status"] == "deferred"
    assert summary["failures"] == []
    assert summary["tiers"][1]["convergence_gate"]["status"] == "pass"
    assert "final tier does not meet production floor" in summary["blockers"][0]


def test_evaluate_promotion_ladder_fails_backend_mismatch(tmp_path: Path) -> None:
    config = {
        "tiers": [
            {
                "name": "bad_backend",
                "resolution": {"Ntheta": 25, "Nzeta": 51, "Nxi": 100, "NL": 4, "Nx": 4},
                "promotions": {
                    "cpu": _write(tmp_path / "cpu.json", _promotion_payload(0.4)),
                    "gpu": _write(tmp_path / "gpu.json", _promotion_payload(0.41)),
                },
            }
        ]
    }

    summary = evaluate_promotion_ladder(config, backend_root_atol=1.0e-6)

    assert summary["status"] == "fail"
    assert any("cpu_gpu" in failure for failure in summary["failures"])


def test_evaluate_promotion_ladder_requires_refinement_beyond_baseline(tmp_path: Path) -> None:
    config = {
        "tiers": [
            {
                "name": "production_baseline",
                "resolution": {"Ntheta": 25, "Nzeta": 51, "Nxi": 100, "NL": 4, "Nx": 4},
                "promotions": {"cpu": _write(tmp_path / "cpu.json", _promotion_payload(0.4))},
            }
        ]
    }

    summary = evaluate_promotion_ladder(config)

    assert summary["status"] == "deferred"
    assert summary["refinement_comparisons"] == 0
    assert summary["tiers"][0]["backend_comparison_gate"]["status"] == "untested"
    assert any("no refinement comparison" in blocker for blocker in summary["blockers"])


def test_evaluate_promotion_ladder_allows_refined_cpu_only_evidence(tmp_path: Path) -> None:
    config = {
        "tiers": [
            {
                "name": "baseline",
                "resolution": {"Ntheta": 23, "Nzeta": 49, "Nxi": 90, "NL": 4, "Nx": 4},
                "promotions": {"cpu": _write(tmp_path / "low_cpu.json", _promotion_payload(0.4))},
            },
            {
                "name": "production",
                "resolution": {"Ntheta": 25, "Nzeta": 51, "Nxi": 100, "NL": 4, "Nx": 4},
                "promotions": {"cpu": _write(tmp_path / "high_cpu.json", _promotion_payload(0.401))},
            },
        ]
    }

    summary = evaluate_promotion_ladder(config)

    assert summary["status"] == "pass"
    assert summary["refinement_comparisons"] == 1
    assert summary["tiers"][1]["convergence_gate"]["resolution_refined"] is True
    assert all(
        tier["backend_comparison_gate"]["status"] == "untested"
        for tier in summary["tiers"]
    )


@pytest.mark.parametrize(
    "second_resolution",
    [
        {"Ntheta": 27, "Nzeta": 53, "Nxi": 102, "NL": 4, "Nx": 4},
        {"Ntheta": 25, "Nzeta": 51, "Nxi": 100, "NL": 4, "Nx": 4},
    ],
    ids=["identical", "coarsened"],
)
def test_evaluate_promotion_ladder_rejects_non_refinement_grids(
    tmp_path: Path, second_resolution: dict[str, int]
) -> None:
    baseline_resolution = {"Ntheta": 27, "Nzeta": 53, "Nxi": 102, "NL": 4, "Nx": 4}
    config = {
        "tiers": [
            {
                "name": "baseline",
                "resolution": baseline_resolution,
                "promotions": {"cpu": _write(tmp_path / "low.json", _promotion_payload(0.4))},
            },
            {
                "name": "candidate",
                "resolution": second_resolution,
                "promotions": {"cpu": _write(tmp_path / "high.json", _promotion_payload(0.4))},
            },
        ]
    }

    summary = evaluate_promotion_ladder(config)

    assert summary["status"] == "deferred"
    assert summary["refinement_comparisons"] == 0
    assert summary["tiers"][1]["convergence_gate"]["resolution_refined"] is False


@pytest.mark.parametrize(
    ("field", "candidate_value"),
    [("r_n", 0.8), ("n_species", 3)],
)
def test_evaluate_promotion_ladder_requires_fixed_physics_across_refinement(
    tmp_path: Path, field: str, candidate_value: float | int
) -> None:
    baseline = {"r_n": 0.5, "n_species": 2}
    candidate = {**baseline, field: candidate_value}
    config = {
        "tiers": [
            {
                "name": "baseline",
                **baseline,
                "resolution": {"Ntheta": 23, "Nzeta": 49, "Nxi": 90, "NL": 4, "Nx": 4},
                "promotions": {"cpu": _write(tmp_path / "low.json", _promotion_payload(0.4))},
            },
            {
                "name": "candidate",
                **candidate,
                "resolution": {"Ntheta": 25, "Nzeta": 51, "Nxi": 100, "NL": 4, "Nx": 4},
                "promotions": {"cpu": _write(tmp_path / "high.json", _promotion_payload(0.4))},
            },
        ]
    }

    summary = evaluate_promotion_ladder(config)

    assert summary["status"] == "deferred"
    assert summary["refinement_comparisons"] == 0
    assert summary["tiers"][1]["convergence_gate"]["resolution_refined"] is True
    assert any(f"{field}=" in blocker for blocker in summary["blockers"])


@pytest.mark.parametrize("name", ["backend_root_atol", "root_drift_atol"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_evaluate_promotion_ladder_rejects_invalid_tolerances(
    tmp_path: Path, name: str, value: float
) -> None:
    config = {
        "tiers": [
            {
                "name": "baseline",
                "resolution": {"Ntheta": 25, "Nzeta": 51, "Nxi": 100, "NL": 4, "Nx": 4},
                "promotions": {"cpu": _write(tmp_path / "cpu.json", _promotion_payload(0.4))},
            }
        ]
    }

    with pytest.raises(ValueError, match="finite and non-negative"):
        evaluate_promotion_ladder(config, **{name: value})


def test_public_ladder_script_writes_summary_and_figures(tmp_path: Path) -> None:
    cpu = _write(tmp_path / "cpu.json", _promotion_payload(0.4136))
    gpu = _write(tmp_path / "gpu.json", _promotion_payload(0.4136000000001))
    config = tmp_path / "ladder_config.json"
    config.write_text(
        json.dumps(
            {
                "tiers": [
                    {
                        "name": "low",
                        "resolution": {"Ntheta": 7, "Nzeta": 7, "Nxi": 5, "NL": 4, "Nx": 4},
                        "promotions": {"cpu": cpu, "gpu": gpu},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    script = _REPO / "examples" / "optimization" / "summarize_finite_beta_electron_root_ladder.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--out-dir",
            str(tmp_path / "out"),
            "--stem",
            "ladder",
        ],
        cwd=_REPO,
        check=True,
        timeout=20,
    )

    payload = json.loads((tmp_path / "out" / "ladder.json").read_text(encoding="utf-8"))
    assert payload["status"] == "deferred"
    assert (tmp_path / "out" / "ladder.png").exists()
    assert (tmp_path / "out" / "ladder.pdf").exists()
