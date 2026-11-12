from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_rows(name: str) -> list[dict[str, object]]:
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "publication_figures"
        / "artifacts"
        / name
    )
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return list(payload["rows"])
    return payload





def test_lhd_full_collisionality_artifact_is_audited() -> None:
    rows = _load_rows("lhd_collisionality_summary.json")
    labels = sorted({str(row["label"]) for row in rows})
    nuprime = sorted({round(float(row["nuprime"]), 6) for row in rows})
    assert labels == ["Fokker-Planck", "PAS"]
    assert len(rows) == 22
    assert nuprime == [
        0.1,
        0.215443,
        0.464159,
        1.0,
        2.154434,
        4.641588,
        9.999998,
        17.782789,
        31.622769,
        56.234121,
        99.999983,
    ]
    by_key = {
        (str(row["label"]), round(float(row["nuprime"]), 6)): row for row in rows
    }
    for nu in nuprime:
        fp = np.asarray(by_key[("Fokker-Planck", nu)]["transport_matrix"], dtype=float)
        pas = np.asarray(by_key[("PAS", nu)]["transport_matrix"], dtype=float)
        assert np.max(np.abs(fp - pas)) > 1.0




def test_w7x_full_collisionality_artifact_is_audited() -> None:
    rows = _load_rows("w7x_collisionality_summary.json")
    labels = sorted({str(row["label"]) for row in rows})
    nuprime = sorted({round(float(row["nuprime"]), 6) for row in rows})
    assert labels == ["Fokker-Planck", "PAS"]
    assert len(rows) == 22
    assert nuprime == [
        0.100003,
        0.21545,
        0.464173,
        1.00003,
        2.1545,
        4.641729,
        10.000301,
        17.783861,
        31.624676,
        56.237514,
        100.00602,
    ]
    by_key = {
        (str(row["label"]), round(float(row["nuprime"]), 6)): row for row in rows
    }
    for nu in nuprime:
        fp = np.asarray(by_key[("Fokker-Planck", nu)]["transport_matrix"], dtype=float)
        pas = np.asarray(by_key[("PAS", nu)]["transport_matrix"], dtype=float)
        assert np.max(np.abs(fp - pas)) > 1.0
