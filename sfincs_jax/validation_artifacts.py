from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


LANDREMAN_2014_URL = "https://doi.org/10.1063/1.4870073"
LANDREMAN_2014_OPEN_PDF = "https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf"


@dataclass(frozen=True)
class CollisionalityRecord:
    """One transport-matrix row from a literature collisionality scan."""

    label: str
    nuprime: float
    transport_matrix: np.ndarray


@dataclass(frozen=True)
class ErSweepRecord:
    """One model/field point from a radial-electric-field trajectory sweep."""

    model: str
    label: str
    er: float
    er_over_eres: float | None
    particle_flux_vm_psi_hat: float
    heat_flux_vm_psi_hat: float
    fsab_flow: float
    fsab_jhat: float
    output_path: str


DEFAULT_PUBLICATION_ARTIFACTS: dict[str, str] = {
    "lhd_collisionality": "lhd_collisionality_summary.json",
    "w7x_collisionality": "w7x_collisionality_summary.json",
    "tokamak_er_sweep": "er_sweep_tokamak_reference_summary.json",
    "stellarator_er_sweep": "er_sweep_stellarator_fast_reference_summary.json",
}

TRANSPORT_ELEMENTS: dict[str, tuple[int, int]] = {
    "L11": (0, 0),
    "L12": (0, 1),
    "L21": (1, 0),
    "L22": (1, 1),
    "L33": (2, 2),
}


def load_collisionality_records(path: Path) -> list[CollisionalityRecord]:
    """Load FP/PAS transport-matrix records from a checked-in summary artifact."""

    payload = json.loads(Path(path).read_text())
    rows = payload["rows"] if isinstance(payload, dict) else payload
    records: list[CollisionalityRecord] = []
    for row in rows:
        records.append(
            CollisionalityRecord(
                label=str(row["label"]),
                nuprime=float(row["nuprime"]),
                transport_matrix=np.asarray(row["transport_matrix"], dtype=np.float64),
            )
        )
    return sorted(records, key=lambda record: (record.label, record.nuprime))


def load_er_sweep_records(path: Path) -> list[ErSweepRecord]:
    """Load trajectory-model sweep records from a checked-in summary artifact."""

    rows = json.loads(Path(path).read_text())
    return [
        ErSweepRecord(
            model=str(row["model"]),
            label=str(row["label"]),
            er=float(row["er"]),
            er_over_eres=None if row.get("er_over_eres") is None else float(row["er_over_eres"]),
            particle_flux_vm_psi_hat=float(row["particle_flux_vm_psi_hat"]),
            heat_flux_vm_psi_hat=float(row["heat_flux_vm_psi_hat"]),
            fsab_flow=float(row["fsab_flow"]),
            fsab_jhat=float(row["fsab_jhat"]),
            output_path=str(row["output_path"]),
        )
        for row in rows
    ]


def collisionality_grid(records: Sequence[CollisionalityRecord]) -> list[float]:
    """Return the sorted normalized-collisionality grid in a scan."""

    return sorted({round(float(record.nuprime), 12) for record in records})


def collisionality_labels(records: Sequence[CollisionalityRecord]) -> list[str]:
    """Return the sorted collision-model labels in a scan."""

    return sorted({record.label for record in records})


def l11_abs_series(records: Sequence[CollisionalityRecord], *, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(nu', |L11|)`` for one collision model."""

    return transport_element_abs_series(records, label=label, element=TRANSPORT_ELEMENTS["L11"])


def transport_element_abs_series(
    records: Sequence[CollisionalityRecord],
    *,
    label: str,
    element: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(nu', |L_ij|)`` for one collision model and matrix element."""

    selected = sorted((record for record in records if record.label == label), key=lambda record: record.nuprime)
    if not selected:
        raise ValueError(f"No collisionality records found for label {label!r}.")
    i, j = (int(element[0]), int(element[1]))
    nuprime = np.asarray([record.nuprime for record in selected], dtype=np.float64)
    values = np.asarray([abs(float(record.transport_matrix[i, j])) for record in selected], dtype=np.float64)
    return nuprime, values


def collisionality_power_law_slope(
    records: Sequence[CollisionalityRecord],
    *,
    label: str,
    element: tuple[int, int],
    n_fit: int = 3,
) -> float:
    """Fit ``|L_ij| ~ (nu')**slope`` on the high-collisionality tail."""

    nuprime, values = transport_element_abs_series(records, label=label, element=element)
    n_fit = int(n_fit)
    if n_fit < 2:
        raise ValueError("n_fit must be at least 2.")
    if nuprime.size < n_fit:
        raise ValueError(f"Need at least {n_fit} records to fit a power-law slope.")
    tail_nu = nuprime[-n_fit:]
    tail_values = np.maximum(values[-n_fit:], np.finfo(float).tiny)
    return float(np.polyfit(np.log(tail_nu), np.log(tail_values), 1)[0])


def fp_pas_l11_separation(records: Sequence[CollisionalityRecord]) -> list[dict[str, float]]:
    """Measure FP/PAS separation in ``L11`` across collisionality.

    The 2014 SFINCS paper uses these scans to show where pitch-angle scattering
    captures the dominant low-collisionality radial-transport physics and where
    momentum conservation matters at higher collisionality.
    """

    by_key = {(record.label, round(float(record.nuprime), 12)): record for record in records}
    rows: list[dict[str, float]] = []
    for nuprime in collisionality_grid(records):
        fp = by_key[("Fokker-Planck", nuprime)]
        pas = by_key[("PAS", nuprime)]
        fp_l11 = float(fp.transport_matrix[0, 0])
        pas_l11 = float(pas.transport_matrix[0, 0])
        abs_delta = abs(fp_l11 - pas_l11)
        rows.append(
            {
                "nuprime": float(nuprime),
                "fp_l11": fp_l11,
                "pas_l11": pas_l11,
                "abs_delta": float(abs_delta),
                "relative_to_fp": float(abs_delta / max(abs(fp_l11), np.finfo(float).tiny)),
            }
        )
    return rows


def er_zero_field_spread(
    records: Sequence[ErSweepRecord],
    *,
    fields: Sequence[str] = (
        "particle_flux_vm_psi_hat",
        "heat_flux_vm_psi_hat",
        "fsab_flow",
        "fsab_jhat",
    ),
) -> dict[str, float]:
    """Return max-min spread across trajectory models at ``E_r = 0``."""

    zero_records = [record for record in records if record.er == 0.0]
    if not zero_records:
        raise ValueError("No E_r=0 records found in trajectory sweep.")
    spreads: dict[str, float] = {}
    for field in fields:
        values = np.asarray([float(getattr(record, field)) for record in zero_records], dtype=np.float64)
        spreads[str(field)] = float(np.max(values) - np.min(values))
    return spreads


def er_nonzero_model_spread(
    records: Sequence[ErSweepRecord],
    *,
    field: str,
) -> dict[str, float]:
    """Return max-min model spread for one diagnostic at each nonzero ``E_r``."""

    spreads: dict[str, float] = {}
    for er in sorted({record.er for record in records if record.er != 0.0}):
        values = np.asarray([float(getattr(record, field)) for record in records if record.er == er], dtype=np.float64)
        spreads[f"{float(er):.12g}"] = float(np.max(values) - np.min(values))
    return spreads


def _summarize_collisionality(records: Sequence[CollisionalityRecord]) -> dict[str, object]:
    separation = fp_pas_l11_separation(records)
    low = separation[0]
    high = separation[-1]
    return {
        "labels": collisionality_labels(records),
        "nuprime": collisionality_grid(records),
        "l11_fp_pas_separation": separation,
        "l11_low_relative_separation": float(low["relative_to_fp"]),
        "l11_high_relative_separation": float(high["relative_to_fp"]),
        "l11_high_to_low_relative_separation_ratio": float(
            high["relative_to_fp"] / max(low["relative_to_fp"], np.finfo(float).tiny)
        ),
    }


def high_collisionality_trend_summary(
    records: Sequence[CollisionalityRecord],
    *,
    n_fit: int = 3,
) -> dict[str, object]:
    """Summarize high-collisionality power-law trends from a corrected scan artifact."""

    slopes: dict[str, dict[str, float]] = {}
    for label in collisionality_labels(records):
        slopes[label] = {
            name: collisionality_power_law_slope(records, label=label, element=element, n_fit=n_fit)
            for name, element in TRANSPORT_ELEMENTS.items()
        }
    pas_l11_l12_positive = all(slopes["PAS"][name] > 0.5 for name in ("L11", "L12"))
    fp_l11_l12_inverse_like = all(slopes["Fokker-Planck"][name] < -0.5 for name in ("L11", "L12"))
    return {
        "n_fit": int(n_fit),
        "nuprime_tail": collisionality_grid(records)[-int(n_fit) :],
        "slopes": slopes,
        "gates": {
            "pas_l11_l12_positive": bool(pas_l11_l12_positive),
            "fp_l11_l12_inverse_like": bool(fp_l11_l12_inverse_like),
        },
        "state": "asymptotic_trend_proxy" if fp_l11_l12_inverse_like else "needs_wider_high_nu_scan",
    }


def _summarize_er_sweep(records: Sequence[ErSweepRecord]) -> dict[str, object]:
    return {
        "models": sorted({record.model for record in records}),
        "er_values": sorted({float(record.er) for record in records}),
        "zero_field_spread": er_zero_field_spread(records),
        "nonzero_fsab_jhat_spread": er_nonzero_model_spread(records, field="fsab_jhat"),
        "nonzero_fsab_flow_spread": er_nonzero_model_spread(records, field="fsab_flow"),
    }


def build_publication_validation_summary(
    *,
    artifact_dir: Path,
    artifacts: Mapping[str, str] = DEFAULT_PUBLICATION_ARTIFACTS,
) -> dict[str, object]:
    """Build a machine-readable summary for the publication validation dashboard."""

    artifact_dir = Path(artifact_dir)
    lhd = load_collisionality_records(artifact_dir / artifacts["lhd_collisionality"])
    w7x = load_collisionality_records(artifact_dir / artifacts["w7x_collisionality"])
    tokamak = load_er_sweep_records(artifact_dir / artifacts["tokamak_er_sweep"])
    stellarator = load_er_sweep_records(artifact_dir / artifacts["stellarator_er_sweep"])
    return {
        "metadata": {
            "schema_version": 1,
            "kind": "publication_validation_dashboard",
            "literature": [LANDREMAN_2014_URL, LANDREMAN_2014_OPEN_PDF],
            "source_artifacts": dict(artifacts),
        },
        "collisionality": {
            "lhd": _summarize_collisionality(lhd),
            "w7x": _summarize_collisionality(w7x),
        },
        "trajectory_sweeps": {
            "tokamak": _summarize_er_sweep(tokamak),
            "stellarator": _summarize_er_sweep(stellarator),
        },
    }


def build_high_collisionality_trend_proxy_summary(
    *,
    artifact_dir: Path,
    artifacts: Mapping[str, str] = DEFAULT_PUBLICATION_ARTIFACTS,
    n_fit: int = 3,
) -> dict[str, object]:
    """Build the high-collisionality trend proxy summary from corrected artifacts."""

    artifact_dir = Path(artifact_dir)
    lhd = load_collisionality_records(artifact_dir / artifacts["lhd_collisionality"])
    w7x = load_collisionality_records(artifact_dir / artifacts["w7x_collisionality"])
    return {
        "metadata": {
            "schema_version": 1,
            "kind": "high_collisionality_trend_proxy",
            "literature": [LANDREMAN_2014_URL, LANDREMAN_2014_OPEN_PDF],
            "source_artifacts": {
                "lhd_collisionality": artifacts["lhd_collisionality"],
                "w7x_collisionality": artifacts["w7x_collisionality"],
            },
            "notes": [
                "The SFINCS 2014 paper states that PAS L11/L12 scale like +nu at high collisionality.",
                "Momentum-conserving FP/model-operator L11/L12 should approach inverse-nu scaling only in the nu' >> 1 limit.",
                "The checked-in scans stop at nu'=10, so this artifact is a trend proxy, not the full Simakov-Helander analytic-limit reproduction.",
            ],
        },
        "cases": {
            "lhd": high_collisionality_trend_summary(lhd, n_fit=n_fit),
            "w7x": high_collisionality_trend_summary(w7x, n_fit=n_fit),
        },
    }
