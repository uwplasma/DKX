from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_rows(name: str) -> list[dict[str, object]]:
    path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "publication_figures"
        / "artifacts"
        / name
    )
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return list(payload["rows"])
    return payload


def test_lhd_collisionality_artifact_has_expected_labels_and_grid() -> None:
    rows = _load_rows("lhd_collisionality_reaudit_fast_summary.json")
    labels = sorted({str(row["label"]) for row in rows})
    nuprime = sorted({round(float(row["nuprime"]), 6) for row in rows})
    assert labels == ["Fokker-Planck", "PAS"]
    assert len(rows) == 8
    assert nuprime == [0.1, 0.464159, 2.154434, 9.999998]


def test_lhd_collisionality_artifact_resolves_fp_pas_separation() -> None:
    rows = _load_rows("lhd_collisionality_reaudit_fast_summary.json")
    by_key = {(str(row["label"]), float(row["nuprime"])): row for row in rows}
    nuprime_values = sorted({float(row["nuprime"]) for row in rows})
    for nuprime in nuprime_values:
        fp = np.asarray(by_key[("Fokker-Planck", nuprime)]["transport_matrix"], dtype=float)
        pas = np.asarray(by_key[("PAS", nuprime)]["transport_matrix"], dtype=float)

        # On the corrected bounded LHD lane, PAS is consistently stronger than
        # FP in the diagonal transport coefficients.
        assert abs(pas[0, 0]) > abs(fp[0, 0])
        assert abs(pas[1, 1]) > abs(fp[1, 1])


def test_lhd_collisionality_artifact_spans_decade_collisionality_ladder() -> None:
    rows = _load_rows("lhd_collisionality_reaudit_fast_summary.json")
    nuprime_values = np.asarray(sorted({float(row["nuprime"]) for row in rows}), dtype=float)
    ratios = nuprime_values[1:] / nuprime_values[:-1]
    assert np.all(ratios > 4.0)


def test_lhd_full_collisionality_artifact_is_audited() -> None:
    rows = _load_rows("lhd_collisionality_summary.json")
    labels = sorted({str(row["label"]) for row in rows})
    nuprime = sorted({round(float(row["nuprime"]), 6) for row in rows})
    assert labels == ["Fokker-Planck", "PAS"]
    assert len(rows) == 14
    assert nuprime == [
        0.1,
        0.215443,
        0.464159,
        1.0,
        2.154434,
        4.641588,
        9.999998,
    ]
    by_key = {
        (str(row["label"]), round(float(row["nuprime"]), 6)): row for row in rows
    }
    for nu in nuprime:
        fp = np.asarray(by_key[("Fokker-Planck", nu)]["transport_matrix"], dtype=float)
        pas = np.asarray(by_key[("PAS", nu)]["transport_matrix"], dtype=float)
        assert np.max(np.abs(fp - pas)) > 1.0


def test_w7x_collisionality_artifact_has_expected_labels_and_grid() -> None:
    rows = _load_rows("w7x_collisionality_reaudit_fast_summary.json")
    labels = sorted({str(row["label"]) for row in rows})
    nuprime = sorted({round(float(row["nuprime"]), 6) for row in rows})
    assert labels == ["Fokker-Planck", "PAS"]
    assert len(rows) == 8
    assert nuprime == [0.100003, 0.464173, 2.1545, 10.000301]


def test_w7x_collisionality_artifact_resolves_fp_pas_separation() -> None:
    rows = _load_rows("w7x_collisionality_reaudit_fast_summary.json")
    by_key = {(str(row["label"]), float(row["nuprime"])): row for row in rows}
    nuprime_values = sorted({float(row["nuprime"]) for row in rows})
    for nuprime in nuprime_values:
        fp = np.asarray(by_key[("Fokker-Planck", nuprime)]["transport_matrix"], dtype=float)
        pas = np.asarray(by_key[("PAS", nuprime)]["transport_matrix"], dtype=float)
        assert abs(pas[0, 0]) > abs(fp[0, 0])
        assert abs(pas[1, 1]) > abs(fp[1, 1])


def test_w7x_full_collisionality_artifact_is_audited() -> None:
    rows = _load_rows("w7x_collisionality_summary.json")
    labels = sorted({str(row["label"]) for row in rows})
    nuprime = sorted({round(float(row["nuprime"]), 6) for row in rows})
    assert labels == ["Fokker-Planck", "PAS"]
    assert len(rows) == 14
    assert nuprime == [
        0.100003,
        0.21545,
        0.464173,
        1.00003,
        2.1545,
        4.641729,
        10.000301,
    ]
    by_key = {
        (str(row["label"]), round(float(row["nuprime"]), 6)): row for row in rows
    }
    for nu in nuprime:
        fp = np.asarray(by_key[("Fokker-Planck", nu)]["transport_matrix"], dtype=float)
        pas = np.asarray(by_key[("PAS", nu)]["transport_matrix"], dtype=float)
        assert np.max(np.abs(fp - pas)) > 1.0
