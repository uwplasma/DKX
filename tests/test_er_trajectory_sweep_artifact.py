from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_records(name: str) -> dict[tuple[str, float], dict[str, float | str | None]]:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "publication_figures"
        / "artifacts"
        / name
    )
    rows = json.loads(path.read_text())
    return {(str(row["model"]), float(row["er"])): row for row in rows}


def test_tokamak_reference_artifact_contains_expected_models_and_field_points() -> None:
    records = _load_records("er_sweep_tokamak_reference_summary.json")
    models = sorted({model for model, _ in records})
    er_values = sorted({er for _, er in records})
    assert models == ["dkes", "full", "partial"]
    assert er_values == [-30.0, 0.0, 30.0]


def test_tokamak_reference_artifact_has_zero_field_agreement() -> None:
    records = _load_records("er_sweep_tokamak_reference_summary.json")
    zero_rows = [records[(model, 0.0)] for model in ("dkes", "partial", "full")]
    for field in (
        "particle_flux_vm_psi_hat",
        "heat_flux_vm_psi_hat",
        "fsab_flow",
        "fsab_jhat",
    ):
        values = np.asarray([float(row[field]) for row in zero_rows], dtype=float)
        assert np.allclose(values, values[0], rtol=0.0, atol=1e-12), field


def test_tokamak_reference_artifact_shows_finite_er_model_separation() -> None:
    records = _load_records("er_sweep_tokamak_reference_summary.json")
    for er in (-30.0, 30.0):
        dkes = records[("dkes", er)]
        partial = records[("partial", er)]
        full = records[("full", er)]

        # The full-trajectory model carries the strongest |FSAB flow| response on
        # the pinned tokamak-like lane, while the partial model produces the
        # strongest particle-flux response away from Er=0.
        fsab = np.asarray(
            [float(dkes["fsab_flow"]), float(partial["fsab_flow"]), float(full["fsab_flow"])],
            dtype=float,
        )
        particle_flux = np.asarray(
            [
                abs(float(dkes["particle_flux_vm_psi_hat"])),
                abs(float(partial["particle_flux_vm_psi_hat"])),
                abs(float(full["particle_flux_vm_psi_hat"])),
            ],
            dtype=float,
        )
        assert abs(float(full["fsab_flow"])) == float(np.max(np.abs(fsab)))
        assert abs(float(partial["particle_flux_vm_psi_hat"])) == float(np.max(particle_flux))
        assert float(np.max(np.abs(fsab)) - np.min(np.abs(fsab))) > 1e-2


def test_stellarator_fast_artifact_contains_expected_models_and_field_points() -> None:
    records = _load_records("er_sweep_stellarator_fast_reference_summary.json")
    models = sorted({model for model, _ in records})
    er_values = sorted({er for _, er in records})
    assert models == ["dkes", "full", "partial"]
    assert er_values == [-8.5897, 0.0, 8.5897]


def test_stellarator_fast_artifact_has_zero_field_agreement() -> None:
    records = _load_records("er_sweep_stellarator_fast_reference_summary.json")
    zero_rows = [records[(model, 0.0)] for model in ("dkes", "partial", "full")]
    for field in (
        "particle_flux_vm_psi_hat",
        "heat_flux_vm_psi_hat",
        "fsab_flow",
        "fsab_jhat",
    ):
        values = np.asarray([float(row[field]) for row in zero_rows], dtype=float)
        assert np.allclose(values, values[0], rtol=0.0, atol=1e-12), field


def test_stellarator_fast_artifact_shows_nonzero_er_spread() -> None:
    records = _load_records("er_sweep_stellarator_fast_reference_summary.json")
    for er in (-8.5897, 8.5897):
        values = np.asarray(
            [
                float(records[("dkes", er)]["fsab_jhat"]),
                float(records[("partial", er)]["fsab_jhat"]),
                float(records[("full", er)]["fsab_jhat"]),
            ],
            dtype=float,
        )
        assert float(np.max(values) - np.min(values)) > 1e-4
