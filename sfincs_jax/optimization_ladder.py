"""Convergence-ladder gates for promoted optimization candidates.

The promotion comparison layer checks one completed scan on several backends.
This module adds the next gate: compare promoted scans across increasing
resolution tiers and fail closed unless the highest tier reaches the declared
production floor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DEFAULT_PRODUCTION_FLOOR = {
    "Ntheta": 25,
    "Nzeta": 51,
    "Nxi": 100,
    "NL": 4,
    "Nx": 4,
}


def load_ladder_config(path: str | Path) -> dict[str, Any]:
    """Load a finite-beta promotion-ladder configuration JSON."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("ladder config JSON must decode to an object")
    return payload


def estimate_rhs1_active_size(
    *,
    ntheta: int,
    nzeta: int,
    nxi: int,
    nx: int,
    n_species: int = 2,
    constraints: int = 4,
) -> int:
    """Estimate RHSMode=1 active unknowns for the finite-beta two-species deck."""

    return int(ntheta) * int(nzeta) * int(nxi) * int(nx) * int(n_species) + int(constraints)


def dense_matrix_gib(active_size: int, *, dtype_bytes: int = 8) -> float:
    """Return the dense matrix memory footprint in GiB."""

    return float(int(active_size) ** 2 * int(dtype_bytes) / (1024.0**3))


def evaluate_promotion_ladder(
    config: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    backend_root_atol: float = 1.0e-6,
    root_drift_atol: float = 2.0e-2,
    production_floor: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Evaluate a promoted optimization convergence ladder.

    The returned summary uses ``status='pass'`` only when all tiers pass their
    backend gates, the final tier reaches the production floor, and adjacent
    tier root drift is below ``root_drift_atol``.  Otherwise a clean but
    under-resolved ladder is reported as ``deferred`` rather than ``fail``.
    """

    root = Path(base_dir).resolve() if base_dir is not None else Path.cwd()
    floor = _normalized_resolution(production_floor or DEFAULT_PRODUCTION_FLOOR)
    raw_tiers = config.get("tiers")
    if not isinstance(raw_tiers, list) or not raw_tiers:
        raise ValueError("ladder config must contain a non-empty 'tiers' list")

    tiers: list[dict[str, Any]] = []
    failures: list[str] = []
    blockers: list[str] = []
    previous_root: float | None = None
    all_backend_clean = True
    all_converged = True

    for index, raw_tier in enumerate(raw_tiers):
        if not isinstance(raw_tier, Mapping):
            raise TypeError("each ladder tier must be an object")
        tier = _evaluate_tier(
            raw_tier,
            base_dir=root,
            production_floor=floor,
            previous_root=previous_root,
            backend_root_atol=float(backend_root_atol),
            root_drift_atol=float(root_drift_atol),
            is_baseline=index == 0,
        )
        tiers.append(tier)
        lane_failures = list(tier.get("failures", []))
        if lane_failures:
            failures.extend(str(item) for item in lane_failures)
            all_backend_clean = False
        if tier["convergence_gate"]["status"] == "fail":
            all_converged = False
            blockers.append(
                f"{tier['name']}: root drift {tier['convergence_gate']['root_drift_from_previous']:.6g} "
                f"exceeds {float(root_drift_atol):.6g}"
            )
        previous_root = _reference_root(tier)

    final_tier = tiers[-1]
    if not final_tier["production_floor_met"]:
        blockers.append(f"{final_tier['name']}: final tier does not meet production floor")

    if failures:
        status = "fail"
    elif final_tier["production_floor_met"] and all_backend_clean and all_converged:
        status = "pass"
    else:
        status = "deferred"

    return {
        "workflow": "sfincs_jax_finite_beta_electron_root_convergence_ladder",
        "status": status,
        "claim_boundary": (
            "This summary compares already-promoted finite-beta QA electron-root "
            "scans across resolution tiers. It is publication-grade only when "
            "the final tier reaches the declared production floor and the "
            "convergence/backend gates pass."
        ),
        "production_floor": floor,
        "tolerances": {
            "backend_root_atol": float(backend_root_atol),
            "root_drift_atol": float(root_drift_atol),
        },
        "tiers": tiers,
        "failures": failures,
        "blockers": blockers,
    }


def _evaluate_tier(
    raw_tier: Mapping[str, Any],
    *,
    base_dir: Path,
    production_floor: Mapping[str, int],
    previous_root: float | None,
    backend_root_atol: float,
    root_drift_atol: float,
    is_baseline: bool,
) -> dict[str, Any]:
    name = str(raw_tier.get("name") or f"tier_{id(raw_tier)}")
    resolution = _normalized_resolution(raw_tier.get("resolution", {}))
    n_species = int(raw_tier.get("n_species", 2))
    active_size = estimate_rhs1_active_size(
        ntheta=resolution["Ntheta"],
        nzeta=resolution["Nzeta"],
        nxi=resolution["Nxi"],
        nx=resolution["Nx"],
        n_species=n_species,
    )
    lanes = _load_lanes(raw_tier.get("promotions", {}), base_dir=base_dir)
    lane_summaries = {lane: _summarize_promotion_payload(payload) for lane, payload in lanes.items()}

    failures: list[str] = []
    for lane, payload in lane_summaries.items():
        if payload["gate_status"] != "pass":
            failures.append(f"{name}:{lane}: promotion gate did not pass")
        if payload["root_type"] != "electron":
            failures.append(f"{name}:{lane}: selected root is not an electron root")
        if not np.isfinite(float(payload["selected_root_er"])):
            failures.append(f"{name}:{lane}: selected root is not finite")

    backend_diffs: dict[str, float] = {}
    if "cpu" in lane_summaries and "gpu" in lane_summaries:
        backend_diffs["cpu_gpu"] = abs(
            float(lane_summaries["cpu"]["selected_root_er"]) - float(lane_summaries["gpu"]["selected_root_er"])
        )
    if "cpu" in lane_summaries and "fortran_v3" in lane_summaries:
        backend_diffs["cpu_fortran_v3"] = abs(
            float(lane_summaries["cpu"]["selected_root_er"])
            - float(lane_summaries["fortran_v3"]["selected_root_er"])
        )
    for pair, diff in backend_diffs.items():
        if diff > float(backend_root_atol):
            failures.append(f"{name}:{pair}: selected root diff {diff:.6g} exceeds {backend_root_atol:.6g}")

    reference_root = _reference_root_from_lanes(lane_summaries)
    if is_baseline or previous_root is None:
        convergence_gate = {
            "status": "baseline",
            "root_drift_from_previous": 0.0,
            "root_drift_atol": float(root_drift_atol),
        }
    else:
        drift = abs(float(reference_root) - float(previous_root))
        convergence_gate = {
            "status": "pass" if drift <= float(root_drift_atol) else "fail",
            "root_drift_from_previous": float(drift),
            "root_drift_atol": float(root_drift_atol),
        }

    return {
        "name": name,
        "resolution": resolution,
        "r_n": float(raw_tier.get("r_n", 0.5)),
        "n_species": n_species,
        "active_size_estimate": int(active_size),
        "dense_matrix_gib": dense_matrix_gib(active_size),
        "production_floor_met": _meets_floor(resolution, production_floor),
        "lanes": lane_summaries,
        "backend_root_diffs": backend_diffs,
        "convergence_gate": convergence_gate,
        "reference_root_er": float(reference_root),
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }


def _normalized_resolution(raw: Mapping[str, Any]) -> dict[str, int]:
    aliases = {
        "Ntheta": ("Ntheta", "ntheta", "n_theta"),
        "Nzeta": ("Nzeta", "nzeta", "n_zeta"),
        "Nxi": ("Nxi", "nxi", "n_xi"),
        "NL": ("NL", "nl", "n_l"),
        "Nx": ("Nx", "nx", "n_x"),
    }
    out: dict[str, int] = {}
    for canonical, keys in aliases.items():
        value = None
        for key in keys:
            if key in raw:
                value = raw[key]
                break
        if value is None:
            if canonical == "NL":
                value = 4
            else:
                raise ValueError(f"resolution is missing {canonical}")
        out[canonical] = int(value)
    return out


def _load_lanes(raw: Any, *, base_dir: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("each ladder tier must define promotion JSON paths")
    lanes: dict[str, dict[str, Any]] = {}
    for lane, raw_path in raw.items():
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = base_dir / path
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError(f"{path} must decode to an object")
        lanes[str(lane)] = payload
    return lanes


def _summarize_promotion_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    selected_root = payload.get("selected_root")
    if not isinstance(selected_root, Mapping):
        selected_root = {}
    flux = payload.get("flux_objective")
    if not isinstance(flux, Mapping):
        flux = {}
    return {
        "gate_status": str(payload.get("gate_status")),
        "selected_root_er": float(selected_root.get("er", np.nan)),
        "root_type": selected_root.get("root_type"),
        "root_bracket": list(selected_root.get("bracket", [])),
        "root_slope": float(selected_root.get("slope", np.nan)),
        "bootstrap_objective": float(payload.get("bootstrap_objective", np.nan)),
        "flux_objective_total": float(flux.get("total", np.nan)),
    }


def _reference_root_from_lanes(lanes: Mapping[str, Mapping[str, Any]]) -> float:
    if "cpu" in lanes:
        return float(lanes["cpu"]["selected_root_er"])
    first = next(iter(lanes.values()))
    return float(first["selected_root_er"])


def _reference_root(tier: Mapping[str, Any]) -> float:
    return float(tier["reference_root_er"])


def _meets_floor(resolution: Mapping[str, int], floor: Mapping[str, int]) -> bool:
    return all(int(resolution[key]) >= int(floor[key]) for key in ("Ntheta", "Nzeta", "Nxi", "NL", "Nx"))


__all__ = [
    "DEFAULT_PRODUCTION_FLOOR",
    "dense_matrix_gib",
    "estimate_rhs1_active_size",
    "evaluate_promotion_ladder",
    "load_ladder_config",
]
