"""Measurement series, the records they come from, and their summaries.

Collisionality and electric-field series, the record types and loaders that
read them, phase timing, and the per-case suite metrics built on top. This is
what the summary builders in ``artifacts`` are built *from*.

Depends only on :mod:`tools.release.lanes`, for the collisionality protocol.
"""

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
from .lanes import CollisionalityLike
from dkx.paths import repository_root


TRANSPORT_ELEMENTS: dict[str, tuple[int, int]] = {
    "L11": (0, 0),
    "L12": (0, 1),
    "L21": (1, 0),
    "L22": (1, 1),
    "L33": (2, 2),
}


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


def collisionality_grid(records: Sequence[CollisionalityLike]) -> list[float]:
    """Return the sorted normalized-collisionality grid in a scan."""

    return sorted({round(float(record.nuprime), 12) for record in records})


def collisionality_labels(records: Sequence[CollisionalityLike]) -> list[str]:
    """Return the sorted collision-model labels in a scan."""

    return sorted({record.label for record in records})


def l11_abs_series(records: Sequence[CollisionalityLike], *, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(nu', |L11|)`` for one collision model."""

    return transport_element_abs_series(records, label=label, element=TRANSPORT_ELEMENTS["L11"])


def transport_element_abs_series(
    records: Sequence[CollisionalityLike],
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
    records: Sequence[CollisionalityLike],
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


def fp_pas_l11_separation(records: Sequence[CollisionalityLike]) -> list[dict[str, float]]:
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


def high_collisionality_trend_summary(
    records: Sequence[CollisionalityLike],
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


def high_collisionality_slope_sensitivity(
    records: Sequence[CollisionalityLike],
    *,
    label: str = "Fokker-Planck",
    elements: Sequence[str] = ("L11", "L12"),
    n_fit_values: Sequence[int] = (2, 3, 4, 5),
) -> list[dict[str, object]]:
    """Return tail-slope fits for several fit-window lengths.

    This is used for the Simakov-Helander audit: a robust high-collisionality
    claim should not depend sensitively on whether the last two, three, or four
    scan points are used for the log-log fit.
    """

    rows: list[dict[str, object]] = []
    max_points = len([record for record in records if record.label == label])
    for n_fit in n_fit_values:
        if int(n_fit) < 2 or int(n_fit) > max_points:
            continue
        slopes = {
            element_name: collisionality_power_law_slope(
                records,
                label=label,
                element=TRANSPORT_ELEMENTS[element_name],
                n_fit=int(n_fit),
            )
            for element_name in elements
        }
        rows.append({"n_fit": int(n_fit), "slopes": slopes})
    return rows


def recommended_high_collisionality_nuprime_grid(
    current_grid: Sequence[float],
    *,
    min_nuprime_for_full_limit: float,
    points_per_decade: int = 4,
) -> list[float]:
    """Recommend additional ``nu'`` values for a full high-collisionality audit.

    The Simakov-Helander comparison is only defensible once the fitted tail is
    clearly in ``nu' >> 1``. This helper converts the current scan extent into a
    compact logarithmic extension that reaches at least one decade past the last
    checked point or the configured full-limit threshold, whichever is larger.
    """

    grid = np.asarray([float(v) for v in current_grid if np.isfinite(float(v)) and float(v) > 0.0], dtype=np.float64)
    if grid.size == 0:
        raise ValueError("current_grid must contain at least one positive finite nuprime value.")
    current_max = float(np.max(grid))
    required = float(min_nuprime_for_full_limit)
    if current_max >= required:
        return []
    target = max(required, 10.0 * current_max)
    n_points = max(2, int(np.ceil((np.log10(target) - np.log10(current_max)) * int(points_per_decade))) + 1)
    values = np.logspace(np.log10(current_max), np.log10(target), n_points)
    extension = [float(v) for v in values if v > current_max * (1.0 + 1.0e-12)]
    if not extension or extension[-1] < target * (1.0 - 1.0e-12):
        extension.append(float(target))
    return extension


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
    repo_root = repository_root() or Path.cwd()
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
    """Load a frozen CPU/GPU suite report from ``python -m tools.release.suite scaled``.

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
