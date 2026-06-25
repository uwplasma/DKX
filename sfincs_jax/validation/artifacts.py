from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field as dataclass_field
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any, Iterator

import numpy as np

from sfincs_jax.validation.math import (
    TRANSPORT_ELEMENTS as TRANSPORT_ELEMENTS,
    collisionality_grid as collisionality_grid,
    collisionality_labels as collisionality_labels,
    collisionality_power_law_slope as collisionality_power_law_slope,
    fp_pas_l11_separation as fp_pas_l11_separation,
    high_collisionality_slope_sensitivity as high_collisionality_slope_sensitivity,
    high_collisionality_trend_summary as high_collisionality_trend_summary,
    l11_abs_series as l11_abs_series,
    recommended_high_collisionality_nuprime_grid as recommended_high_collisionality_nuprime_grid,
    transport_element_abs_series as transport_element_abs_series,
)


LANDREMAN_2014_URL = "https://doi.org/10.1063/1.4870077"
LANDREMAN_2014_OPEN_PDF = "https://publications.lib.chalmers.se/records/fulltext/199559/local_199559.pdf"
SFINCS_FORTRAN_REPO_URL = "https://github.com/landreman/sfincs"
SIMAKOV_HELANDER_HIGH_COLLISIONALITY_URL = "https://doi.org/10.1063/1.3104715"
PAUL_2019_ADJOINT_URL = "https://arxiv.org/abs/1904.06430"
SFINCS_ADJOINT_APS_URL = "https://meetings-archive.aps.org/dpp/2018/bp11/36/"

PUBLIC_3D_BENCHMARK_FLOOR = {"NTHETA": 25, "NZETA": 51, "NX": 4, "NXI": 100}
PUBLIC_TOKAMAK_BENCHMARK_FLOOR = {"NTHETA": 25, "NZETA": 1, "NX": 4, "NXI": 100}
FORTRAN_SUITE_BENCHMARK_SCHEMA_VERSION = 1
FORTRAN_SUITE_BENCHMARK_KIND = "fortran_v3_suite_benchmark_summary"
FORTRAN_SUITE_BENCHMARK_REPORT_KEYS = (
    "total_cases",
    "parity_ok_cases",
    "jax_error_cases",
    "max_attempts_cases",
    "strict_mismatch_total",
    "runtime_ratio_summary",
    "warm_or_logged_runtime_ratio_summary",
    "memory_ratio_summary",
    "active_memory_ratio_summary",
)

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


def maxrss_mb(*, platform: str = sys.platform, raw_value: int | None = None) -> float:
    """Return process maximum resident set size in MB."""

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if raw_value is None else int(raw_value)
    if str(platform).startswith("darwin"):
        return float(raw) / (1024.0 * 1024.0)
    return float(raw) / 1024.0


@dataclass
class PhaseRecord:
    """One timed phase in a benchmark or audit run."""

    name: str
    elapsed_s: float
    status: str = "ok"
    maxrss_mb: float | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class PhaseTimer:
    """Collect bounded phase timings for JSON run reports."""

    def __init__(self) -> None:
        self._start_s = time.perf_counter()
        self.records: list[PhaseRecord] = []

    @contextlib.contextmanager
    def phase(self, name: str, **metadata: Any) -> Iterator[None]:
        start_s = time.perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            self.records.append(
                PhaseRecord(
                    name=name,
                    elapsed_s=round(max(0.0, time.perf_counter() - start_s), 6),
                    status=status,
                    maxrss_mb=round(maxrss_mb(), 6),
                    metadata=dict(metadata),
                )
            )

    def summary(self) -> dict[str, Any]:
        elapsed = max(0.0, time.perf_counter() - self._start_s)
        return {
            "elapsed_s": round(elapsed, 6),
            "maxrss_mb": round(maxrss_mb(), 6),
            "phase_count": len(self.records),
            "phases": [record.to_json() for record in self.records],
        }


def _repo_stable_path(path: Path) -> str:
    """Return a reproducible path for checked-in validation metadata."""

    path = Path(path)
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


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
    jax_runtime_s_cold: float | None
    jax_runtime_s_warm: float | None
    jax_logged_elapsed_s: float | None
    fortran_max_rss_mb: float | None
    jax_max_rss_mb: float | None
    jax_incremental_max_rss_mb: float | None
    jax_rss_baseline_mb: float | None
    jax_memory_metric_source: str | None
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
    def cold_runtime_ratio(self) -> float | None:
        """Return cold external JAX runtime divided by Fortran runtime when available."""

        return _safe_ratio(self.jax_runtime_s_cold, self.fortran_runtime_s)

    @property
    def warm_runtime_ratio(self) -> float | None:
        """Return warm JAX rerun runtime divided by Fortran runtime when available."""

        return _safe_ratio(self.jax_runtime_s_warm, self.fortran_runtime_s)

    @property
    def warm_or_logged_runtime_s(self) -> float | None:
        """Return warm rerun runtime, falling back to logged CLI elapsed time."""

        return self.jax_runtime_s_warm if self.jax_runtime_s_warm is not None else self.jax_logged_elapsed_s

    @property
    def warm_or_logged_runtime_source(self) -> str | None:
        """Return the source field used for the warm-runtime comparison plot."""

        if self.jax_runtime_s_warm is not None:
            return "jax_runtime_s_warm"
        if self.jax_logged_elapsed_s is not None:
            return "jax_logged_elapsed_s"
        return None

    @property
    def warm_or_logged_runtime_ratio(self) -> float | None:
        """Return warm-rerun-or-logged JAX elapsed time divided by Fortran runtime."""

        return _safe_ratio(self.warm_or_logged_runtime_s, self.fortran_runtime_s)

    @property
    def memory_ratio(self) -> float | None:
        """Return ``jax_max_rss_mb / fortran_max_rss_mb`` when both values are finite."""

        return _safe_ratio(self.jax_max_rss_mb, self.fortran_max_rss_mb)

    @property
    def active_jax_memory_mb(self) -> float | None:
        """Return profiler-derived active JAX memory, falling back to process RSS.

        ``jax_max_rss_mb`` remains the external-command process high-water mark.
        The active value subtracts the fixed Python/JAX/XLA runtime baseline when
        profiler ``dpeak_rss_mb`` or ``drss_mb`` data are available, which is the
        fairer solver-memory metric for public per-case bars.
        """

        return self.jax_incremental_max_rss_mb if self.jax_incremental_max_rss_mb is not None else self.jax_max_rss_mb

    @property
    def active_memory_ratio(self) -> float | None:
        """Return active JAX memory divided by Fortran process RSS."""

        return _safe_ratio(self.active_jax_memory_mb, self.fortran_max_rss_mb)


DEFAULT_PUBLICATION_ARTIFACTS: dict[str, str] = {
    "lhd_collisionality": "lhd_collisionality_summary.json",
    "w7x_collisionality": "w7x_collisionality_summary.json",
    "tokamak_er_sweep": "er_sweep_tokamak_reference_summary.json",
    "stellarator_er_sweep": "er_sweep_stellarator_fast_reference_summary.json",
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


def load_autodiff_sensitivity_summary(path: Path) -> Mapping[str, object]:
    """Load a checked-in autodiff/sensitivity validation summary artifact."""

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"Autodiff summary {path} must contain a JSON object.")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("kind") != "autodiff_sensitivity_validation":
        raise ValueError(f"Autodiff summary {path} has an unexpected metadata.kind.")
    return payload


def autodiff_gradient_error_summary(payload: Mapping[str, object]) -> dict[str, float | int]:
    """Summarize finite-difference agreement from an autodiff validation payload."""

    checks = payload.get("gradient_checks", [])
    if not isinstance(checks, Sequence):
        raise ValueError("gradient_checks must be a sequence.")
    rel_errors: list[float] = []
    abs_errors: list[float] = []
    for check in checks:
        if not isinstance(check, Mapping):
            continue
        rel_error = _optional_float(check.get("relative_error"))
        abs_error = _optional_float(check.get("absolute_error"))
        if rel_error is not None:
            rel_errors.append(rel_error)
        if abs_error is not None:
            abs_errors.append(abs_error)
    return {
        "count": int(len(rel_errors)),
        "max_relative_error": float(max(rel_errors)) if rel_errors else float("nan"),
        "median_relative_error": float(np.median(rel_errors)) if rel_errors else float("nan"),
        "max_absolute_error": float(max(abs_errors)) if abs_errors else float("nan"),
    }


def build_autodiff_sensitivity_validation_summary(
    *,
    gradient_checks: Sequence[Mapping[str, object]],
    finite_difference_sweep: Sequence[Mapping[str, object]],
    geometry_sensitivity: Mapping[str, object],
    cost_scaling: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object] | None = None,
    relative_error_gate: float = 1.0e-4,
    residual_gate: float = 1.0e-8,
) -> dict[str, object]:
    """Build the machine-readable summary for the autodiff validation figure lane."""

    gradient_rows = [dict(row) for row in gradient_checks]
    fd_rows = [dict(row) for row in finite_difference_sweep]
    cost_rows = [dict(row) for row in cost_scaling]
    meta = dict(metadata or {})
    meta.setdefault("schema_version", 1)
    meta.setdefault("kind", "autodiff_sensitivity_validation")
    meta.setdefault("literature", [PAUL_2019_ADJOINT_URL, SFINCS_ADJOINT_APS_URL])
    meta.setdefault(
        "notes",
        [
            "Gradients through the linear solve use jax.lax.custom_linear_solve.",
            "The validation checks implicit differentiation against centered finite differences.",
            "The geometry map is a differentiable Boozer-harmonic sensitivity scaffold, not a full VMEC boundary optimization claim.",
        ],
    )
    payload: dict[str, object] = {
        "metadata": meta,
        "gradient_checks": gradient_rows,
        "finite_difference_sweep": fd_rows,
        "geometry_sensitivity": dict(geometry_sensitivity),
        "cost_scaling": cost_rows,
    }
    err = autodiff_gradient_error_summary(payload)
    residuals = [
        _optional_float(row.get("primal_residual_norm"))
        for row in gradient_rows
        if _optional_float(row.get("primal_residual_norm")) is not None
    ]
    adjoint_residuals = [
        _optional_float(row.get("adjoint_residual_norm"))
        for row in gradient_rows
        if _optional_float(row.get("adjoint_residual_norm")) is not None
    ]
    max_residual = max(residuals) if residuals else float("nan")
    max_adjoint_residual = max(adjoint_residuals) if adjoint_residuals else float("nan")
    max_rel = float(err["max_relative_error"])
    payload["gates"] = {
        "relative_error_gate": float(relative_error_gate),
        "residual_gate": float(residual_gate),
        "max_relative_error": max_rel,
        "max_primal_residual_norm": float(max_residual),
        "max_adjoint_residual_norm": float(max_adjoint_residual),
        "gradient_relative_error_ok": bool(np.isfinite(max_rel) and max_rel <= float(relative_error_gate)),
        "primal_residual_ok": bool(np.isfinite(max_residual) and max_residual <= float(residual_gate)),
        "adjoint_residual_ok": bool(
            not adjoint_residuals or (np.isfinite(max_adjoint_residual) and max_adjoint_residual <= float(residual_gate))
        ),
    }
    payload["gradient_error_summary"] = err
    return payload


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
                jax_runtime_s_cold=_optional_float(row.get("jax_runtime_s_cold", row.get("jax_runtime_s"))),
                jax_runtime_s_warm=_optional_float(row.get("jax_runtime_s_warm")),
                jax_logged_elapsed_s=_optional_float(row.get("jax_logged_elapsed_s")),
                fortran_max_rss_mb=_optional_float(row.get("fortran_max_rss_mb")),
                jax_max_rss_mb=_optional_float(row.get("jax_max_rss_mb")),
                jax_incremental_max_rss_mb=_optional_float(row.get("jax_incremental_max_rss_mb")),
                jax_rss_baseline_mb=_optional_float(row.get("jax_rss_baseline_mb")),
                jax_memory_metric_source=(
                    None if row.get("jax_memory_metric_source") is None else str(row.get("jax_memory_metric_source"))
                ),
                practical_mismatches=_mismatch_count(row, SUITE_MISMATCH_FIELDS),
                strict_mismatches=_mismatch_count(row, SUITE_STRICT_MISMATCH_FIELDS),
            )
        )
    return sorted(metrics, key=lambda item: item.case)


def filter_suite_metrics_by_fortran_runtime(
    cpu_metrics: Sequence[SuiteCaseMetric],
    gpu_metrics: Sequence[SuiteCaseMetric],
    *,
    min_fortran_runtime_s: float | None,
) -> tuple[list[SuiteCaseMetric], list[SuiteCaseMetric], list[dict[str, object]]]:
    """Filter CPU/GPU benchmark rows to cases with a sufficiently large reference run.

    Very small Fortran runs are useful as CI parity checks, but they are poor public
    performance comparisons because filesystem, process-launch, and JIT amortization
    dominate the wall clock. The reference runtime is taken from the CPU report when
    present, falling back to the GPU report only for GPU-only cases.
    """

    cpu_by_case = {metric.case: metric for metric in cpu_metrics}
    gpu_by_case = {metric.case: metric for metric in gpu_metrics}
    if min_fortran_runtime_s is None:
        return sorted(cpu_metrics, key=lambda item: item.case), sorted(gpu_metrics, key=lambda item: item.case), []

    threshold = float(min_fortran_runtime_s)
    included_cases: set[str] = set()
    excluded_cases: list[dict[str, object]] = []
    for case in sorted(set(cpu_by_case) | set(gpu_by_case)):
        reference_metric = cpu_by_case.get(case) or gpu_by_case.get(case)
        runtime = reference_metric.fortran_runtime_s if reference_metric is not None else None
        if runtime is not None and np.isfinite(float(runtime)) and float(runtime) >= threshold:
            included_cases.add(case)
        else:
            excluded_cases.append(
                {
                    "case": case,
                    "fortran_runtime_s": None if runtime is None else float(runtime),
                }
            )

    filtered_cpu = sorted((metric for metric in cpu_metrics if metric.case in included_cases), key=lambda item: item.case)
    filtered_gpu = sorted((metric for metric in gpu_metrics if metric.case in included_cases), key=lambda item: item.case)
    return filtered_cpu, filtered_gpu, excluded_cases


def _row_resolution(row: Mapping[str, object]) -> dict[str, int] | None:
    """Return normalized resolution metadata from a suite row when available."""

    raw = row.get("final_resolution") or row.get("benchmark_resolution")
    if not isinstance(raw, Mapping):
        return None
    resolution: dict[str, int] = {}
    for key, value in raw.items():
        try:
            resolution[str(key).upper()] = int(value)
        except (TypeError, ValueError):
            continue
    return resolution or None


def _row_floor(row: Mapping[str, object]) -> dict[str, int]:
    """Return the public benchmark floor appropriate for one suite row."""

    case = str(row.get("case", "")).lower()
    resolution = _row_resolution(row) or {}
    n_zeta = resolution.get("NZETA")
    if "tokamak" in case or n_zeta == 1:
        return PUBLIC_TOKAMAK_BENCHMARK_FLOOR
    return PUBLIC_3D_BENCHMARK_FLOOR


def benchmark_resolution_floor_violations(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Return public benchmark rows that are below the production-resolution floor."""

    violations: list[dict[str, object]] = []
    for row in rows:
        case = str(row.get("case", ""))
        resolution = _row_resolution(row)
        if resolution is None:
            violations.append(
                {
                    "case": case,
                    "reason": "missing_final_resolution",
                    "resolution": None,
                    "required": _row_floor(row),
                }
            )
            continue
        floor = _row_floor(row)
        missing_or_low = {
            key: {"actual": resolution.get(key), "required": int(required)}
            for key, required in floor.items()
            if resolution.get(key) is None or int(resolution[key]) < int(required)
        }
        if missing_or_low:
            violations.append(
                {
                    "case": case,
                    "reason": "below_public_benchmark_resolution_floor",
                    "resolution": dict(sorted(resolution.items())),
                    "required": dict(floor),
                    "fields": missing_or_low,
                }
            )
    return violations


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
        "jax_runtime_s_cold": metric.jax_runtime_s_cold,
        "jax_runtime_s_warm": metric.jax_runtime_s_warm,
        "jax_logged_elapsed_s": metric.jax_logged_elapsed_s,
        "warm_or_logged_runtime_s": metric.warm_or_logged_runtime_s,
        "warm_or_logged_runtime_source": metric.warm_or_logged_runtime_source,
        "fortran_max_rss_mb": metric.fortran_max_rss_mb,
        "jax_max_rss_mb": metric.jax_max_rss_mb,
        "jax_incremental_max_rss_mb": metric.jax_incremental_max_rss_mb,
        "jax_rss_baseline_mb": metric.jax_rss_baseline_mb,
        "jax_memory_metric_source": metric.jax_memory_metric_source,
        "active_jax_memory_mb": metric.active_jax_memory_mb,
        "runtime_ratio": metric.runtime_ratio,
        "cold_runtime_ratio": metric.cold_runtime_ratio,
        "warm_runtime_ratio": metric.warm_runtime_ratio,
        "warm_or_logged_runtime_ratio": metric.warm_or_logged_runtime_ratio,
        "logged_runtime_ratio": metric.logged_runtime_ratio,
        "memory_ratio": metric.memory_ratio,
        "active_memory_ratio": metric.active_memory_ratio,
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
        "cold_runtime_ratio_summary": _ratio_summary([metric.cold_runtime_ratio for metric in metrics]),
        "warm_runtime_ratio_summary": _ratio_summary([metric.warm_runtime_ratio for metric in metrics]),
        "warm_or_logged_runtime_ratio_summary": _ratio_summary(
            [metric.warm_or_logged_runtime_ratio for metric in metrics]
        ),
        "warm_or_logged_runtime_source_counts": _counts(
            [metric.warm_or_logged_runtime_source or "missing" for metric in metrics]
        ),
        "logged_runtime_ratio_summary": _ratio_summary([metric.logged_runtime_ratio for metric in metrics]),
        "memory_ratio_summary": _ratio_summary([metric.memory_ratio for metric in metrics]),
        "active_memory_ratio_summary": _ratio_summary([metric.active_memory_ratio for metric in metrics]),
        "fastest_jax_vs_fortran_cases": _top_metrics(metrics, key="runtime_ratio", n=n_top, reverse=False),
        "slowest_jax_vs_fortran_cases": _top_metrics(metrics, key="runtime_ratio", n=n_top, reverse=True),
        "highest_memory_ratio_cases": _top_metrics(metrics, key="memory_ratio", n=n_top, reverse=True),
        "highest_active_memory_ratio_cases": _top_metrics(
            metrics,
            key="active_memory_ratio",
            n=n_top,
            reverse=True,
        ),
        "highest_jax_runtime_cases": _top_metrics(metrics, key="jax_runtime_s", n=n_top, reverse=True),
        "highest_jax_memory_cases": _top_metrics(metrics, key="jax_max_rss_mb", n=n_top, reverse=True),
        "highest_active_jax_memory_cases": _top_metrics(
            metrics,
            key="active_jax_memory_mb",
            n=n_top,
            reverse=True,
        ),
    }


def fortran_suite_benchmark_schema_errors(payload: Mapping[str, object]) -> list[str]:
    """Return schema-contract errors for README/docs benchmark summaries."""

    errors: list[str] = []
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return ["missing metadata mapping"]
    if metadata.get("kind") != FORTRAN_SUITE_BENCHMARK_KIND:
        errors.append("metadata.kind must be fortran_v3_suite_benchmark_summary")
    if metadata.get("schema_version") != FORTRAN_SUITE_BENCHMARK_SCHEMA_VERSION:
        errors.append(f"metadata.schema_version must be {FORTRAN_SUITE_BENCHMARK_SCHEMA_VERSION}")
    reports = payload.get("reports")
    if not isinstance(reports, Mapping):
        errors.append("missing reports mapping")
        return errors
    for backend in ("cpu", "gpu"):
        report = reports.get(backend)
        if not isinstance(report, Mapping):
            errors.append(f"reports.{backend} must be a mapping")
            continue
        for key in FORTRAN_SUITE_BENCHMARK_REPORT_KEYS:
            if key not in report:
                errors.append(f"reports.{backend}.{key} missing")
    return errors


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


def _periodic_central_derivative(values: np.ndarray, coordinates: np.ndarray, *, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.size < 2:
        return np.zeros_like(values)
    spacing = float(coordinates[1] - coordinates[0])
    if not np.isfinite(spacing) or spacing == 0.0:
        raise ValueError("Periodic derivative coordinates must have finite nonzero spacing.")
    return (np.roll(values, -1, axis=int(axis)) - np.roll(values, 1, axis=int(axis))) / (2.0 * spacing)


def _theta_zeta_axes(shape: tuple[int, ...], *, n_theta: int, n_zeta: int) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"Expected a two-dimensional geometry field, got shape {shape}.")
    if shape == (int(n_theta), int(n_zeta)):
        return 0, 1
    if shape == (int(n_zeta), int(n_theta)):
        return 1, 0
    raise ValueError(
        f"Geometry field shape {shape} does not match theta/zeta sizes {(int(n_theta), int(n_zeta))}."
    )


def appendix_b_geometry_audit_from_h5(output_h5: Path) -> dict[str, object]:
    """Compute discrete Appendix-B geometry ingredients from a SFINCS output file.

    The returned coefficients are a normalization audit, not a final validation
    claim. They use the same checked-in geometry fields that appear in
    ``sfincsOutput.h5`` and make the Simakov-Helander/Pfirsch-Schluter comparison
    reproducible enough to identify which high-collisionality scans are still
    missing before an analytic-limit overlay is promoted.
    """

    try:
        import h5py
    except Exception as exc:  # pragma: no cover - h5py is a package dependency.
        raise RuntimeError("appendix_b_geometry_audit_from_h5 requires h5py.") from exc

    output_h5 = Path(output_h5)
    required = (
        "BHat",
        "DHat",
        "uHat",
        "BHat_sup_theta",
        "BHat_sup_zeta",
        "dBHatdtheta",
        "dBHatdzeta",
        "theta",
        "zeta",
        "GHat",
        "IHat",
        "iota",
        "FSABHat2",
    )
    with h5py.File(output_h5, "r") as h5:
        missing = [name for name in required if name not in h5]
        if missing:
            raise ValueError(f"{output_h5} is missing Appendix-B audit fields: {missing}")
        b_hat = np.asarray(h5["BHat"], dtype=np.float64)
        d_hat = np.asarray(h5["DHat"], dtype=np.float64)
        u_hat = np.asarray(h5["uHat"], dtype=np.float64)
        b_sup_theta = np.asarray(h5["BHat_sup_theta"], dtype=np.float64)
        b_sup_zeta = np.asarray(h5["BHat_sup_zeta"], dtype=np.float64)
        db_dtheta = np.asarray(h5["dBHatdtheta"], dtype=np.float64)
        db_dzeta = np.asarray(h5["dBHatdzeta"], dtype=np.float64)
        theta = np.asarray(h5["theta"], dtype=np.float64)
        zeta = np.asarray(h5["zeta"], dtype=np.float64)
        g_hat = float(np.asarray(h5["GHat"], dtype=np.float64))
        i_hat = float(np.asarray(h5["IHat"], dtype=np.float64))
        iota = float(np.asarray(h5["iota"], dtype=np.float64))
        fsab_hat2 = float(np.asarray(h5["FSABHat2"], dtype=np.float64))

    theta_axis, zeta_axis = _theta_zeta_axes(b_hat.shape, n_theta=theta.size, n_zeta=zeta.size)
    if np.any(d_hat == 0.0):
        raise ValueError(f"{output_h5} contains zero DHat entries.")
    weights = 1.0 / d_hat
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        raise ValueError(f"{output_h5} has invalid flux-surface-average weights.")

    def fsa(quantity: np.ndarray) -> float:
        return float(np.sum(weights * np.asarray(quantity, dtype=np.float64)) / weight_sum)

    def grad_parallel(quantity: np.ndarray) -> np.ndarray:
        dtheta = _periodic_central_derivative(quantity, theta, axis=theta_axis)
        dzeta = _periodic_central_derivative(quantity, zeta, axis=zeta_axis)
        return (b_sup_theta * dtheta + b_sup_zeta * dzeta) / b_hat

    gradpar_b = (b_sup_theta * db_dtheta + b_sup_zeta * db_dzeta) / b_hat
    gradpar_ln_b = gradpar_b / b_hat
    gradpar_u_b2 = grad_parallel(u_hat * b_hat * b_hat)
    gradpar_b2_fsa = fsa(gradpar_b * gradpar_b)
    if abs(gradpar_b2_fsa) <= np.finfo(float).tiny:
        raise ValueError(f"{output_h5} has a near-zero <(grad_parallel B)^2> denominator.")

    fsa_b2 = fsa(b_hat * b_hat)
    fsa_u_b2 = fsa(u_hat * b_hat * b_hat)
    fsa_u2_b2 = fsa(u_hat * u_hat * b_hat * b_hat)
    fsa_gradlnb_gradu_b2 = fsa(gradpar_ln_b * gradpar_u_b2)
    fsa_u_gradpar_b2 = fsa(u_hat * gradpar_b * gradpar_b)
    g1 = (fsa_gradlnb_gradu_b2 * fsa_gradlnb_gradu_b2) / gradpar_b2_fsa - fsa(
        (gradpar_u_b2 / b_hat) ** 2
    )
    g2 = fsa(u_hat * gradpar_ln_b * gradpar_u_b2) - fsa_gradlnb_gradu_b2 * fsa_u_gradpar_b2 / gradpar_b2_fsa
    k1 = fsa_gradlnb_gradu_b2 / (2.0 * gradpar_b2_fsa)
    k2 = 1.97213 * fsa_u_b2 / fsa_b2 - 1.03287 * 2.0 * k1 + 0.09361 * fsa_u_gradpar_b2 / gradpar_b2_fsa
    h_geom = (fsa_u_b2 * fsa_u_b2) / fsa_b2 - fsa_u2_b2
    g_plus_iota_i = g_hat + iota * i_hat
    common = 0.96 * np.sqrt(2.0) * (g_plus_iota_i**2) / (iota * iota * g_hat * g_hat)
    coefficients = {
        "L11": float(common * 0.75 * g1),
        "L12": float(common * (3.245 * g1 + 0.085 * g2)),
        "L22": float(np.sqrt(2.0) * 8.0 / 5.0 * fsa_b2 * h_geom / (iota * iota * g_hat * g_hat)),
        "L33": float(fsa_b2 * fsa_b2 / (3.0 * 0.96 * np.sqrt(2.0) * (g_plus_iota_i**2) * gradpar_b2_fsa)),
    }
    return {
        "source_output": str(output_h5),
        "grid": {
            "n_theta": int(theta.size),
            "n_zeta": int(zeta.size),
            "theta_axis": int(theta_axis),
            "zeta_axis": int(zeta_axis),
        },
        "geometry_scalars": {
            "GHat": float(g_hat),
            "IHat": float(i_hat),
            "iota": float(iota),
            "G_plus_iota_I": float(g_plus_iota_i),
            "FSABHat2_output": float(fsab_hat2),
            "FSABHat2_recomputed": float(fsa_b2),
            "FSABHat2_relative_error": float(abs(fsa_b2 - fsab_hat2) / max(abs(fsab_hat2), np.finfo(float).tiny)),
        },
        "appendix_b_discrete_quantities": {
            "G1": float(g1),
            "G2": float(g2),
            "K1": float(k1),
            "K2": float(k2),
            "H": float(h_geom),
            "gradpar_b_rms": float(np.sqrt(abs(gradpar_b2_fsa))),
            "fsa_u_b2": float(fsa_u_b2),
            "fsa_u2_b2": float(fsa_u2_b2),
        },
        "transport_matrix_coefficients_over_nuprime": coefficients,
        "notes": [
            "Coefficients follow the Appendix-B structure using checked-in normalized sfincs_jax output fields.",
            "Use these values as an audit of normalization and geometry ingredients, not as a final analytic-limit acceptance gate.",
        ],
    }


def _inverse_tail_ratio(
    records: Sequence[CollisionalityRecord],
    *,
    label: str,
    element: tuple[int, int],
    coefficient_over_nuprime: float,
) -> dict[str, float]:
    nuprime, values = transport_element_abs_series(records, label=label, element=element)
    last_nu = float(nuprime[-1])
    last_value = float(values[-1])
    predicted = abs(float(coefficient_over_nuprime)) / max(last_nu, np.finfo(float).tiny)
    return {
        "nuprime": last_nu,
        "observed_abs": last_value,
        "appendix_b_proxy_abs": float(predicted),
        "observed_to_proxy_ratio": float(last_value / max(predicted, np.finfo(float).tiny)),
    }


def _simakov_case_summary(
    records: Sequence[CollisionalityRecord],
    *,
    geometry_audit: Mapping[str, object] | None,
    n_fit: int,
    min_nuprime_for_full_limit: float,
    target_slope: float,
    slope_tolerance: float,
) -> dict[str, object]:
    trend = high_collisionality_trend_summary(records, n_fit=n_fit)
    sensitivity = high_collisionality_slope_sensitivity(records, n_fit_values=(2, 3, 4, 5))
    slopes = trend["slopes"]["Fokker-Planck"]  # type: ignore[index]
    fp_l11_l12_target_like = all(
        abs(float(slopes[name]) - float(target_slope)) <= float(slope_tolerance) for name in ("L11", "L12")
    )
    grid = collisionality_grid(records)
    max_nuprime = float(max(grid))
    scan_extends_to_required_high_nu = max_nuprime >= float(min_nuprime_for_full_limit)
    high_nu_extension = recommended_high_collisionality_nuprime_grid(
        grid,
        min_nuprime_for_full_limit=float(min_nuprime_for_full_limit),
    )
    appendix_ratios: dict[str, object] = {}
    if geometry_audit is not None:
        coeffs = geometry_audit.get("transport_matrix_coefficients_over_nuprime", {})
        if isinstance(coeffs, Mapping):
            for name in ("L11", "L12", "L22", "L33"):
                if name in coeffs:
                    appendix_ratios[name] = _inverse_tail_ratio(
                        records,
                        label="Fokker-Planck",
                        element=TRANSPORT_ELEMENTS[name],
                        coefficient_over_nuprime=float(coeffs[name]),
                    )
    return {
        "nuprime_grid": grid,
        "max_nuprime": max_nuprime,
        "recommended_high_nuprime_extension": high_nu_extension,
        "trend": trend,
        "slope_sensitivity": sensitivity,
        "appendix_b_geometry_audit": dict(geometry_audit) if geometry_audit is not None else None,
        "appendix_b_proxy_ratios_at_max_nuprime": appendix_ratios,
        "gates": {
            "scan_extends_to_required_high_nu": bool(scan_extends_to_required_high_nu),
            "fp_l11_l12_target_inverse_slope": bool(fp_l11_l12_target_like),
            "pas_l11_l12_positive": bool(trend["gates"]["pas_l11_l12_positive"]),  # type: ignore[index]
            "appendix_b_geometry_inputs_available": bool(geometry_audit is not None),
        },
        "state": "ready_for_full_overlay" if scan_extends_to_required_high_nu and fp_l11_l12_target_like else "needs_wider_high_nu_scan",
    }


def build_simakov_helander_limit_audit_summary(
    *,
    artifact_dir: Path,
    artifacts: Mapping[str, str] = DEFAULT_PUBLICATION_ARTIFACTS,
    geometry_outputs: Mapping[str, Path] | None = None,
    precomputed_geometry_audits: Mapping[str, Mapping[str, object]] | None = None,
    n_fit: int = 3,
    min_nuprime_for_full_limit: float = 50.0,
    target_slope: float = -1.0,
    slope_tolerance: float = 0.35,
) -> dict[str, object]:
    """Build the bounded audit for the Simakov-Helander high-collisionality lane."""

    artifact_dir = Path(artifact_dir)
    lhd = load_collisionality_records(artifact_dir / artifacts["lhd_collisionality"])
    w7x = load_collisionality_records(artifact_dir / artifacts["w7x_collisionality"])
    geometry_audits: dict[str, Mapping[str, object] | None] = {"lhd": None, "w7x": None}
    if precomputed_geometry_audits is not None:
        for case in ("lhd", "w7x"):
            audit = precomputed_geometry_audits.get(case)
            if audit is not None:
                geometry_audits[case] = dict(audit)
    if geometry_outputs is not None:
        for case in ("lhd", "w7x"):
            path = geometry_outputs.get(case)
            if path is not None and Path(path).exists():
                geometry_audits[case] = appendix_b_geometry_audit_from_h5(Path(path))

    cases = {
        "lhd": _simakov_case_summary(
            lhd,
            geometry_audit=geometry_audits["lhd"],
            n_fit=n_fit,
            min_nuprime_for_full_limit=min_nuprime_for_full_limit,
            target_slope=target_slope,
            slope_tolerance=slope_tolerance,
        ),
        "w7x": _simakov_case_summary(
            w7x,
            geometry_audit=geometry_audits["w7x"],
            n_fit=n_fit,
            min_nuprime_for_full_limit=min_nuprime_for_full_limit,
            target_slope=target_slope,
            slope_tolerance=slope_tolerance,
        ),
    }
    full_ready = all(bool(case["state"] == "ready_for_full_overlay") for case in cases.values())
    geometry_ready = all(bool(case["gates"]["appendix_b_geometry_inputs_available"]) for case in cases.values())  # type: ignore[index]
    literature_ready = bool(full_ready and geometry_ready)
    return {
        "metadata": {
            "schema_version": 1,
            "kind": "simakov_helander_limit_audit",
            "literature": [
                LANDREMAN_2014_URL,
                LANDREMAN_2014_OPEN_PDF,
                SIMAKOV_HELANDER_HIGH_COLLISIONALITY_URL,
            ],
            "source_artifacts": {
                "lhd_collisionality": artifacts["lhd_collisionality"],
                "w7x_collisionality": artifacts["w7x_collisionality"],
            },
            "notes": [
                "This artifact audits the normalization and high-nu sufficiency for the Appendix-B analytic-limit lane.",
                "It intentionally keeps the full reproduction gate closed until a wider nu' >> 1 scan is checked in.",
                "The current full collisionality summaries stop near nu'=10, below the default full-limit threshold.",
            ],
            "publication_figure": {
                "claim_status": (
                    "checked_in_converged_artifact" if literature_ready else "proxy_or_deferred"
                ),
                "artifact_class": (
                    "checked_in_simakov_helander_full_limit_artifact"
                    if literature_ready
                    else "checked_in_normalization_audit_deferred_full_limit"
                ),
                "checked_in_converged_artifact": bool(literature_ready),
                "ready_for_physics_validation_claim": bool(literature_ready),
                "manuscript_label": (
                    "checked-in Simakov-Helander full high-nu validation"
                    if literature_ready
                    else "normalization audit; full Simakov-Helander high-nu validation deferred"
                ),
            },
        },
        "configuration": {
            "n_fit": int(n_fit),
            "min_nuprime_for_full_limit": float(min_nuprime_for_full_limit),
            "target_fp_slope": float(target_slope),
            "slope_tolerance": float(slope_tolerance),
        },
        "cases": cases,
        "gates": {
            "appendix_b_geometry_inputs_available": bool(geometry_ready),
            "all_cases_ready_for_full_overlay": bool(full_ready),
            "checked_in_converged_artifact": bool(literature_ready),
            "ready_for_literature_claim": bool(literature_ready),
            "proxy_or_deferred_only": bool(not literature_ready),
            "full_simakov_helander_reproduction_closed": bool(not literature_ready),
        },
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
    min_fortran_runtime_s: float | None = None,
    enforce_public_resolution_floor: bool = True,
) -> dict[str, object]:
    """Build a CPU/GPU suite benchmark summary against the Fortran v3 reference."""

    cpu_report = Path(cpu_report)
    gpu_report = Path(gpu_report)
    raw_cpu_rows = load_suite_report(cpu_report)
    raw_gpu_rows = load_suite_report(gpu_report)
    raw_cpu_metrics = suite_case_metrics(raw_cpu_rows)
    raw_gpu_metrics = suite_case_metrics(raw_gpu_rows)
    cpu_metrics, gpu_metrics, excluded_cases = filter_suite_metrics_by_fortran_runtime(
        raw_cpu_metrics,
        raw_gpu_metrics,
        min_fortran_runtime_s=min_fortran_runtime_s,
    )
    cpu_reported_cases = {metric.case for metric in cpu_metrics}
    gpu_reported_cases = {metric.case for metric in gpu_metrics}
    cpu_rows = [row for row in raw_cpu_rows if str(row.get("case", "")) in cpu_reported_cases]
    gpu_rows = [row for row in raw_gpu_rows if str(row.get("case", "")) in gpu_reported_cases]
    resolution_floor_violations = {
        "cpu": benchmark_resolution_floor_violations(cpu_rows),
        "gpu": benchmark_resolution_floor_violations(gpu_rows),
    }
    if enforce_public_resolution_floor and (
        resolution_floor_violations["cpu"] or resolution_floor_violations["gpu"]
    ):
        raise ValueError(
            "Public benchmark summary includes below-floor or untagged rows: "
            + json.dumps(resolution_floor_violations, sort_keys=True)
        )
    payload = {
        "metadata": {
            "schema_version": FORTRAN_SUITE_BENCHMARK_SCHEMA_VERSION,
            "kind": FORTRAN_SUITE_BENCHMARK_KIND,
            "literature": [
                LANDREMAN_2014_URL,
                LANDREMAN_2014_OPEN_PDF,
                SFINCS_FORTRAN_REPO_URL,
            ],
            "source_reports": {
                "cpu": _repo_stable_path(cpu_report),
                "gpu": _repo_stable_path(gpu_report),
            },
            "source_case_counts": {
                "cpu": int(len(raw_cpu_metrics)),
                "gpu": int(len(raw_gpu_metrics)),
            },
            "reported_case_counts": {
                "cpu": int(len(cpu_metrics)),
                "gpu": int(len(gpu_metrics)),
            },
            "min_fortran_runtime_s": None if min_fortran_runtime_s is None else float(min_fortran_runtime_s),
            "excluded_low_fortran_runtime_cases": excluded_cases,
            "public_3d_benchmark_floor": dict(PUBLIC_3D_BENCHMARK_FLOOR),
            "public_tokamak_benchmark_floor": dict(PUBLIC_TOKAMAK_BENCHMARK_FLOOR),
            "resolution_floor_violations": resolution_floor_violations,
            "notes": [
                "Runtime ratios use audited wall-clock fields stored in the frozen suite reports.",
                "Process memory ratios use audited maximum-RSS fields; active JAX memory ratios use profiler dpeak_rss_mb/drss_mb deltas when available.",
                "The summary is a release gate: all audited CPU/GPU cases must remain parity_ok with no strict mismatches before filtering.",
                "README-facing performance plots filter out very short Fortran reference runs so public runtime claims are based on production-scale rows.",
                "README-facing performance plots also require final_resolution metadata meeting the public production-resolution floor.",
                "The artifacts compare sfincs_jax against the Fortran v3 reference implementation on the vendored example suite.",
            ],
        },
        "reports": {
            "cpu": suite_report_summary(cpu_rows, label="CPU"),
            "gpu": suite_report_summary(gpu_rows, label="GPU"),
        },
    }
    schema_errors = fortran_suite_benchmark_schema_errors(payload)
    if schema_errors:
        raise ValueError("Invalid Fortran-suite benchmark summary schema: " + "; ".join(schema_errors))
    return payload


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
    cases = {
        "lhd": high_collisionality_trend_summary(lhd, n_fit=n_fit),
        "w7x": high_collisionality_trend_summary(w7x, n_fit=n_fit),
    }
    all_pas_positive = all(
        bool(case["gates"]["pas_l11_l12_positive"])  # type: ignore[index]
        for case in cases.values()
    )
    all_fp_inverse_like = all(
        bool(case["gates"]["fp_l11_l12_inverse_like"])  # type: ignore[index]
        for case in cases.values()
    )
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
            "publication_figure": {
                "claim_status": "proxy_or_deferred",
                "artifact_class": "checked_in_high_collisionality_trend_proxy",
                "checked_in_converged_artifact": False,
                "ready_for_physics_validation_claim": False,
                "manuscript_label": "checked-in trend proxy; full analytic-limit validation deferred",
            },
        },
        "cases": cases,
        "gates": {
            "all_pas_l11_l12_positive": bool(all_pas_positive),
            "all_fp_l11_l12_inverse_like": bool(all_fp_inverse_like),
            "checked_in_converged_artifact": False,
            "ready_for_literature_claim": False,
            "full_simakov_helander_reproduction_closed": True,
            "proxy_or_deferred_only": True,
        },
    }
