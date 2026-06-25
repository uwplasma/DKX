"""Promotion-summary comparison helpers for optimization evidence.

The helpers in this module compare already-computed promotion JSON payloads.
They do not run SFINCS solves; they only gate the summary quantities needed to
claim CPU/GPU agreement and optional Fortran-v3 agreement.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any


PromotionPayloadInput = Mapping[str, Any] | str | Path


@dataclass(frozen=True)
class PromotionComparisonTolerances:
    """Numeric gates for optimization promotion comparisons."""

    selected_root_er_rtol: float = 1.0e-7
    selected_root_er_atol: float = 1.0e-10
    bootstrap_objective_rtol: float = 1.0e-6
    flux_objective_total_rtol: float = 1.0e-6


_MISSING = object()

_FIELD_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "selected_root_er": (
        ("selected_root", "er"),
        ("selected_root_er",),
        ("ambipolar_root", "er"),
        ("root", "er"),
    ),
    "bootstrap_objective": (
        ("bootstrap_objective",),
        ("objectives", "bootstrap_objective"),
        ("objectives", "bootstrap"),
        ("metrics", "bootstrap_objective", "value"),
    ),
    "flux_objective_total": (
        ("flux_objective", "total"),
        ("flux_objective_total",),
        ("objectives", "flux_objective_total"),
        ("objectives", "flux_total"),
        ("metrics", "flux_objective_total", "value"),
    ),
    "gate_status": (
        ("gate_status",),
        ("promotion_gate", "status"),
    ),
}


def load_promotion_payload(payload: PromotionPayloadInput) -> dict[str, Any]:
    """Return a promotion payload from a mapping, path, or raw JSON string."""

    if isinstance(payload, Mapping):
        return dict(payload)
    if isinstance(payload, Path):
        loaded = json.loads(payload.read_text(encoding="utf-8"))
    else:
        text = str(payload).strip()
        if text.startswith("{"):
            loaded = json.loads(text)
        else:
            loaded = json.loads(Path(text).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError("promotion payload JSON must decode to an object")
    return loaded


def compare_promotion_pair(
    reference_payload: PromotionPayloadInput,
    candidate_payload: PromotionPayloadInput,
    *,
    reference_label: str,
    candidate_label: str,
    tolerances: PromotionComparisonTolerances | None = None,
    require_gate_pass: bool = True,
    require_flux_objective: bool = True,
) -> dict[str, Any]:
    """Compare two optimization promotion summaries.

    The returned dictionary is intended to be JSON-serializable and stable for
    CI/reporting consumers: top-level ``status``, ``failures``, and ``metrics``
    carry all gate outcomes.
    """

    tol = PromotionComparisonTolerances() if tolerances is None else tolerances
    reference = load_promotion_payload(reference_payload)
    candidate = load_promotion_payload(candidate_payload)
    failures: list[str] = []
    metrics: dict[str, dict[str, Any]] = {}

    metrics["gate_status"] = _compare_gate_status(
        reference,
        candidate,
        reference_label=reference_label,
        candidate_label=candidate_label,
        failures=failures,
        require_gate_pass=require_gate_pass,
    )
    metrics["selected_root_er"] = _compare_numeric_field(
        "selected_root_er",
        reference,
        candidate,
        reference_label=reference_label,
        candidate_label=candidate_label,
        rtol=tol.selected_root_er_rtol,
        atol=tol.selected_root_er_atol,
        failures=failures,
    )
    metrics["bootstrap_objective"] = _compare_numeric_field(
        "bootstrap_objective",
        reference,
        candidate,
        reference_label=reference_label,
        candidate_label=candidate_label,
        rtol=tol.bootstrap_objective_rtol,
        atol=0.0,
        failures=failures,
    )
    if require_flux_objective or _has_field(reference, "flux_objective_total") or _has_field(
        candidate,
        "flux_objective_total",
    ):
        metrics["flux_objective_total"] = _compare_numeric_field(
            "flux_objective_total",
            reference,
            candidate,
            reference_label=reference_label,
            candidate_label=candidate_label,
            rtol=tol.flux_objective_total_rtol,
            atol=0.0,
            failures=failures,
        )

    return {
        "comparison": f"{reference_label}_vs_{candidate_label}_promotion",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "metrics": metrics,
        "tolerances": asdict(tol),
        "reference_label": reference_label,
        "candidate_label": candidate_label,
    }


def compare_cpu_gpu_promotions(
    cpu_payload: PromotionPayloadInput,
    gpu_payload: PromotionPayloadInput,
    *,
    tolerances: PromotionComparisonTolerances | None = None,
    require_gate_pass: bool = True,
    require_flux_objective: bool = True,
) -> dict[str, Any]:
    """Compare CPU and GPU promotion-summary payloads."""

    return compare_promotion_pair(
        cpu_payload,
        gpu_payload,
        reference_label="cpu",
        candidate_label="gpu",
        tolerances=tolerances,
        require_gate_pass=require_gate_pass,
        require_flux_objective=require_flux_objective,
    )


def compare_fortran_promotion(
    sfincs_jax_payload: PromotionPayloadInput,
    fortran_v3_payload: PromotionPayloadInput,
    *,
    tolerances: PromotionComparisonTolerances | None = None,
    require_gate_pass: bool = True,
    require_flux_objective: bool = True,
    sfincs_jax_label: str = "sfincs_jax",
    fortran_label: str = "fortran_v3",
) -> dict[str, Any]:
    """Compare a SFINCS_JAX promotion payload with a Fortran-v3-derived one."""

    return compare_promotion_pair(
        sfincs_jax_payload,
        fortran_v3_payload,
        reference_label=sfincs_jax_label,
        candidate_label=fortran_label,
        tolerances=tolerances,
        require_gate_pass=require_gate_pass,
        require_flux_objective=require_flux_objective,
    )


def compare_optimization_promotions(
    cpu_payload: PromotionPayloadInput,
    gpu_payload: PromotionPayloadInput,
    *,
    fortran_v3_payload: PromotionPayloadInput | None = None,
    sfincs_jax_fortran_payload: PromotionPayloadInput | None = None,
    tolerances: PromotionComparisonTolerances | None = None,
    require_gate_pass: bool = True,
    require_flux_objective: bool = True,
) -> dict[str, Any]:
    """Compare CPU/GPU promotions and optionally SFINCS_JAX against Fortran v3.

    When ``fortran_v3_payload`` is provided, the Fortran comparison uses
    ``sfincs_jax_fortran_payload`` as its JAX-side reference if supplied;
    otherwise the CPU payload is used.
    """

    tol = PromotionComparisonTolerances() if tolerances is None else tolerances
    comparisons = {
        "cpu_gpu": compare_cpu_gpu_promotions(
            cpu_payload,
            gpu_payload,
            tolerances=tol,
            require_gate_pass=require_gate_pass,
            require_flux_objective=require_flux_objective,
        )
    }
    if fortran_v3_payload is not None:
        jax_reference = (
            cpu_payload if sfincs_jax_fortran_payload is None else sfincs_jax_fortran_payload
        )
        comparisons["sfincs_jax_fortran_v3"] = compare_fortran_promotion(
            jax_reference,
            fortran_v3_payload,
            tolerances=tol,
            require_gate_pass=require_gate_pass,
            require_flux_objective=require_flux_objective,
        )

    failures: list[str] = []
    for comparison_name, comparison in comparisons.items():
        for failure in comparison["failures"]:
            failures.append(f"{comparison_name}: {failure}")

    return {
        "workflow": "sfincs_jax_optimization_promotion_comparison",
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "comparisons": comparisons,
        "tolerances": asdict(tol),
    }


def _lookup(payload: Mapping[str, Any], field: str) -> Any:
    for path in _FIELD_PATHS[field]:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                current = _MISSING
                break
            current = current[key]
        if current is not _MISSING:
            return current
    return _MISSING


def _has_field(payload: Mapping[str, Any], field: str) -> bool:
    return _lookup(payload, field) is not _MISSING


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _relative_difference(reference: float, candidate: float) -> float:
    diff = abs(candidate - reference)
    scale = abs(reference)
    if scale == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / scale


def _compare_gate_status(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    reference_label: str,
    candidate_label: str,
    failures: list[str],
    require_gate_pass: bool,
) -> dict[str, Any]:
    reference_status = _lookup(reference, "gate_status")
    candidate_status = _lookup(candidate, "gate_status")
    reference_text = None if reference_status is _MISSING else str(reference_status).lower()
    candidate_text = None if candidate_status is _MISSING else str(candidate_status).lower()
    metric = {
        "reference": reference_text,
        "candidate": candidate_text,
        "required": "pass" if require_gate_pass else None,
        "status": "pass",
    }
    if require_gate_pass:
        if reference_text != "pass":
            failures.append(f"{reference_label} gate_status is {reference_text!r}; expected 'pass'")
            metric["status"] = "fail"
        if candidate_text != "pass":
            failures.append(f"{candidate_label} gate_status is {candidate_text!r}; expected 'pass'")
            metric["status"] = "fail"
    return metric


def _compare_numeric_field(
    field: str,
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    reference_label: str,
    candidate_label: str,
    rtol: float,
    atol: float,
    failures: list[str],
) -> dict[str, Any]:
    reference_raw = _lookup(reference, field)
    candidate_raw = _lookup(candidate, field)
    reference_value = None if reference_raw is _MISSING else _finite_float(reference_raw)
    candidate_value = None if candidate_raw is _MISSING else _finite_float(candidate_raw)
    metric: dict[str, Any] = {
        "reference": reference_value,
        "candidate": candidate_value,
        "abs_diff": None,
        "rel_diff": None,
        "rtol": float(rtol),
        "atol": float(atol),
        "limit": None,
        "status": "fail",
    }
    if reference_value is None:
        failures.append(f"{reference_label} is missing finite {field}")
        return metric
    if candidate_value is None:
        failures.append(f"{candidate_label} is missing finite {field}")
        return metric

    abs_diff = abs(candidate_value - reference_value)
    rel_diff = _relative_difference(reference_value, candidate_value)
    limit = float(atol) + float(rtol) * abs(reference_value)
    passed = abs_diff <= limit
    metric.update(
        {
            "abs_diff": float(abs_diff),
            "rel_diff": float(rel_diff),
            "limit": float(limit),
            "status": "pass" if passed else "fail",
        }
    )
    if not passed:
        failures.append(
            f"{field} differs between {reference_label} and {candidate_label}: "
            f"abs_diff={abs_diff:.6e} exceeds limit={limit:.6e}"
        )
    return metric


__all__ = [
    "PromotionComparisonTolerances",
    "compare_cpu_gpu_promotions",
    "compare_fortran_promotion",
    "compare_optimization_promotions",
    "compare_promotion_pair",
    "load_promotion_payload",
]
