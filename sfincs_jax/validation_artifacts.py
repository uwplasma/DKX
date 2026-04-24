from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


LANDREMAN_2014_URL = "https://doi.org/10.1063/1.4870077"
LANDREMAN_2014_OPEN_PDF = "https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf"
SFINCS_FORTRAN_REPO_URL = "https://github.com/landreman/sfincs"

SUITE_MISMATCH_FIELDS = (
    "n_mismatch_common",
    "n_mismatch_physics",
    "n_mismatch_solver",
)

SUITE_STRICT_MISMATCH_FIELDS = (
    "strict_n_mismatch_common",
    "strict_n_mismatch_physics",
    "strict_n_mismatch_solver",
)


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


@dataclass(frozen=True)
class SuiteCaseMetric:
    """Runtime, memory, and parity metrics for one audited example-suite case."""

    case: str
    status: str
    blocker_type: str
    fortran_runtime_s: float | None
    jax_runtime_s: float | None
    jax_logged_elapsed_s: float | None
    fortran_max_rss_mb: float | None
    jax_max_rss_mb: float | None
    practical_mismatches: int
    strict_mismatches: int

    @property
    def runtime_ratio(self) -> float | None:
        """Return ``jax_runtime_s / fortran_runtime_s`` when both values are finite."""

        return _safe_ratio(self.jax_runtime_s, self.fortran_runtime_s)

    @property
    def logged_runtime_ratio(self) -> float | None:
        """Return logged JAX elapsed time divided by Fortran runtime when available."""

        return _safe_ratio(self.jax_logged_elapsed_s, self.fortran_runtime_s)

    @property
    def memory_ratio(self) -> float | None:
        """Return ``jax_max_rss_mb / fortran_max_rss_mb`` when both values are finite."""

        return _safe_ratio(self.jax_max_rss_mb, self.fortran_max_rss_mb)


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


def load_suite_report(path: Path) -> list[Mapping[str, object]]:
    """Load a frozen CPU/GPU suite report from ``scripts/run_scaled_example_suite.py``.

    The release-facing report is a list of per-case dictionaries. Some archived
    summary artifacts wrap that list in a top-level ``rows`` key, so this loader
    accepts both layouts while rejecting anything else.
    """

    payload = json.loads(Path(path).read_text())
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"Suite report {path} must contain a list of case rows.")
    return [row for row in rows if isinstance(row, Mapping)]


def collisionality_grid(records: Sequence[CollisionalityRecord]) -> list[float]:
    """Return the sorted normalized-collisionality grid in a scan."""

    return sorted({round(float(record.nuprime), 12) for record in records})


def collisionality_labels(records: Sequence[CollisionalityRecord]) -> list[str]:
    """Return the sorted collision-model labels in a scan."""

    return sorted({record.label for record in records})


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _mismatch_count(row: Mapping[str, object], fields: Sequence[str]) -> int:
    count = 0
    for field in fields:
        try:
            count += int(row.get(field, 0) or 0)
        except (TypeError, ValueError):
            count += 0
    return int(count)


def suite_case_metrics(rows: Sequence[Mapping[str, object]]) -> list[SuiteCaseMetric]:
    """Normalize raw suite rows into typed benchmark/parity metrics."""

    metrics: list[SuiteCaseMetric] = []
    for row in rows:
        metrics.append(
            SuiteCaseMetric(
                case=str(row.get("case", "")),
                status=str(row.get("status", "")),
                blocker_type=str(row.get("blocker_type", "none")),
                fortran_runtime_s=_optional_float(row.get("fortran_runtime_s")),
                jax_runtime_s=_optional_float(row.get("jax_runtime_s")),
                jax_logged_elapsed_s=_optional_float(row.get("jax_logged_elapsed_s")),
                fortran_max_rss_mb=_optional_float(row.get("fortran_max_rss_mb")),
                jax_max_rss_mb=_optional_float(row.get("jax_max_rss_mb")),
                practical_mismatches=_mismatch_count(row, SUITE_MISMATCH_FIELDS),
                strict_mismatches=_mismatch_count(row, SUITE_STRICT_MISMATCH_FIELDS),
            )
        )
    return sorted(metrics, key=lambda item: item.case)


def _counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _ratio_summary(values: Sequence[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def _metric_row(metric: SuiteCaseMetric, *, field: str) -> dict[str, object]:
    return {
        "case": metric.case,
        "status": metric.status,
        "blocker_type": metric.blocker_type,
        "fortran_runtime_s": metric.fortran_runtime_s,
        "jax_runtime_s": metric.jax_runtime_s,
        "jax_logged_elapsed_s": metric.jax_logged_elapsed_s,
        "fortran_max_rss_mb": metric.fortran_max_rss_mb,
        "jax_max_rss_mb": metric.jax_max_rss_mb,
        "runtime_ratio": metric.runtime_ratio,
        "logged_runtime_ratio": metric.logged_runtime_ratio,
        "memory_ratio": metric.memory_ratio,
        "practical_mismatches": metric.practical_mismatches,
        "strict_mismatches": metric.strict_mismatches,
        "sort_field": field,
    }


def _top_metrics(
    metrics: Sequence[SuiteCaseMetric],
    *,
    key: str,
    n: int = 5,
    reverse: bool = True,
) -> list[dict[str, object]]:
    keyed: list[tuple[float, SuiteCaseMetric]] = []
    for metric in metrics:
        value = getattr(metric, key)
        if value is None or not np.isfinite(float(value)):
            continue
        keyed.append((float(value), metric))
    keyed.sort(key=lambda item: item[0], reverse=reverse)
    return [_metric_row(metric, field=key) for _, metric in keyed[: int(n)]]


def suite_report_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    label: str,
    n_top: int = 5,
) -> dict[str, object]:
    """Summarize one frozen suite report for release and manuscript dashboards."""

    metrics = suite_case_metrics(rows)
    statuses = [metric.status for metric in metrics]
    blocker_types = [metric.blocker_type for metric in metrics]
    practical_totals = [metric.practical_mismatches for metric in metrics]
    strict_totals = [metric.strict_mismatches for metric in metrics]
    return {
        "label": str(label),
        "total_cases": int(len(metrics)),
        "status_counts": _counts(statuses),
        "blocker_counts": _counts(blocker_types),
        "parity_ok_cases": int(sum(status == "parity_ok" for status in statuses)),
        "jax_error_cases": int(sum(status == "jax_error" or blocker == "jax_error" for status, blocker in zip(statuses, blocker_types))),
        "max_attempts_cases": int(
            sum(status == "max_attempts" or blocker == "max_attempts" for status, blocker in zip(statuses, blocker_types))
        ),
        "practical_mismatch_cases": int(sum(count > 0 for count in practical_totals)),
        "practical_mismatch_total": int(sum(practical_totals)),
        "strict_mismatch_cases": int(sum(count > 0 for count in strict_totals)),
        "strict_mismatch_total": int(sum(strict_totals)),
        "runtime_ratio_summary": _ratio_summary([metric.runtime_ratio for metric in metrics]),
        "logged_runtime_ratio_summary": _ratio_summary([metric.logged_runtime_ratio for metric in metrics]),
        "memory_ratio_summary": _ratio_summary([metric.memory_ratio for metric in metrics]),
        "fastest_jax_vs_fortran_cases": _top_metrics(metrics, key="runtime_ratio", n=n_top, reverse=False),
        "slowest_jax_vs_fortran_cases": _top_metrics(metrics, key="runtime_ratio", n=n_top, reverse=True),
        "highest_memory_ratio_cases": _top_metrics(metrics, key="memory_ratio", n=n_top, reverse=True),
        "highest_jax_runtime_cases": _top_metrics(metrics, key="jax_runtime_s", n=n_top, reverse=True),
        "highest_jax_memory_cases": _top_metrics(metrics, key="jax_max_rss_mb", n=n_top, reverse=True),
    }


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


def build_fortran_suite_benchmark_summary(
    *,
    cpu_report: Path,
    gpu_report: Path,
) -> dict[str, object]:
    """Build a CPU/GPU suite benchmark summary against the Fortran v3 reference."""

    cpu_report = Path(cpu_report)
    gpu_report = Path(gpu_report)
    return {
        "metadata": {
            "schema_version": 1,
            "kind": "fortran_v3_suite_benchmark_summary",
            "literature": [
                LANDREMAN_2014_URL,
                LANDREMAN_2014_OPEN_PDF,
                SFINCS_FORTRAN_REPO_URL,
            ],
            "source_reports": {
                "cpu": str(cpu_report),
                "gpu": str(gpu_report),
            },
            "notes": [
                "Ratios use the audited wall-clock and maximum-RSS fields stored in the frozen suite reports.",
                "The summary is a release gate: all audited CPU/GPU cases must remain parity_ok with no strict mismatches.",
                "The artifacts compare sfincs_jax against the Fortran v3 reference implementation on the vendored example suite.",
            ],
        },
        "reports": {
            "cpu": suite_report_summary(load_suite_report(cpu_report), label="CPU"),
            "gpu": suite_report_summary(load_suite_report(gpu_report), label="GPU"),
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
