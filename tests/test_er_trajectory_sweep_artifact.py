from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_records() -> dict[tuple[str, float], dict[str, float | str | None]]:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "publication_figures"
        / "artifacts"
        / "er_sweep_fast_tokamak_summary.json"
    )
    rows = json.loads(path.read_text())
    return {(str(row["model"]), float(row["er"])): row for row in rows}


def test_er_sweep_artifact_contains_expected_models_and_field_points() -> None:
    records = _load_records()
    models = sorted({model for model, _ in records})
    er_values = sorted({er for _, er in records})
    assert models == ["dkes", "full", "partial"]
    assert er_values == [-0.5, 0.0, 0.5]


def test_er_sweep_artifact_has_small_field_agreement_at_zero_er() -> None:
    records = _load_records()
    zero_rows = [records[(model, 0.0)] for model in ("dkes", "partial", "full")]
    for field in (
        "particle_flux_vm_psi_hat",
        "heat_flux_vm_psi_hat",
        "fsab_flow",
        "fsab_jhat",
    ):
        values = np.asarray([float(row[field]) for row in zero_rows], dtype=float)
        assert np.allclose(values, values[0], rtol=0.0, atol=1e-12), field


def test_er_sweep_artifact_shows_model_separation_away_from_zero_er() -> None:
    records = _load_records()
    for er in (-0.5, 0.5):
        dkes = records[("dkes", er)]
        partial = records[("partial", er)]
        full = records[("full", er)]

        # On this bounded tokamak-like prototype lane, the partial model carries
        # the largest FSAB flow correction, DKES is intermediate, and the full
        # model is smallest once |Er| moves away from zero.
        assert float(partial["fsab_flow"]) > float(dkes["fsab_flow"]) > float(full["fsab_flow"])

        # Heat-flux separation is also resolved on the same fast lane. The
        # partial model is largest for both signs, while the DKES/full ordering
        # can flip with the sign of Er, so only the resolved spread is locked
        # down here.
        heat_fluxes = np.asarray(
            [
                float(dkes["heat_flux_vm_psi_hat"]),
                float(partial["heat_flux_vm_psi_hat"]),
                float(full["heat_flux_vm_psi_hat"]),
            ],
            dtype=float,
        )
        assert float(partial["heat_flux_vm_psi_hat"]) == float(np.max(heat_fluxes))
        assert float(np.max(heat_fluxes) - np.min(heat_fluxes)) > 1e-10
