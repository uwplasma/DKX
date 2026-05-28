"""Workflow helpers that connect optimization proxies to SFINCS scan gates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CandidateScanPlan:
    """Command plan for promoting one optimization candidate with ``scan-er``."""

    proxy_summary: Path
    input_namelist: Path
    out_dir: Path
    er_values: tuple[float, ...]
    compute_solution: bool
    compute_transport_matrix: bool
    jobs: int
    skip_existing: bool
    scan_command: tuple[str, ...]
    promotion_command: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": "sfincs_jax_optimization_candidate_scan_plan",
            "proxy_summary": str(self.proxy_summary),
            "input_namelist": str(self.input_namelist),
            "out_dir": str(self.out_dir),
            "er_values": [float(value) for value in self.er_values],
            "compute_solution": bool(self.compute_solution),
            "compute_transport_matrix": bool(self.compute_transport_matrix),
            "jobs": int(self.jobs),
            "skip_existing": bool(self.skip_existing),
            "scan_command": list(self.scan_command),
            "scan_command_string": shlex.join(self.scan_command),
            "promotion_command": list(self.promotion_command),
            "promotion_command_string": shlex.join(self.promotion_command),
            "claim_boundary": (
                "This plan launches high-fidelity sfincs_jax scan-er runs for an "
                "accepted optimization candidate. Promotion still requires the "
                "scan outputs to pass residual, ambipolar-root, flux, and "
                "comparison gates."
            ),
        }


def load_proxy_summary(path: str | Path) -> dict[str, Any]:
    """Load a proxy optimization summary JSON."""

    proxy_path = Path(path).resolve()
    if not proxy_path.exists():
        raise FileNotFoundError(str(proxy_path))
    return json.loads(proxy_path.read_text(encoding="utf-8"))


def er_values_from_bounds(*, er_min: float, er_max: float, n: int) -> tuple[float, ...]:
    """Return deterministic Er scan values including both endpoints."""

    if int(n) < 2:
        raise ValueError("n must be >= 2")
    values = np.linspace(float(er_min), float(er_max), int(n), dtype=np.float64)
    return tuple(float(value) for value in values)


def build_candidate_scan_plan(
    *,
    proxy_summary: str | Path,
    input_namelist: str | Path,
    out_dir: str | Path,
    er_values: tuple[float, ...],
    compute_solution: bool = True,
    compute_transport_matrix: bool = False,
    jobs: int = 1,
    skip_existing: bool = True,
    promotion_stem: str = "candidate_promotion",
    require_electron_root: bool = True,
    impurity_species_index: int | None = None,
    target_impurity_flux: float = 0.0,
) -> CandidateScanPlan:
    """Build scan and promotion commands for an accepted candidate."""

    proxy_path = Path(proxy_summary).resolve()
    input_path = Path(input_namelist).resolve()
    scan_dir = Path(out_dir).resolve()
    values = tuple(float(value) for value in er_values)
    if len(values) < 2:
        raise ValueError("at least two Er values are required")
    if int(jobs) < 1:
        raise ValueError("jobs must be >= 1")

    scan_cmd = [
        sys.executable,
        "-m",
        "sfincs_jax",
        "scan-er",
        "--input",
        str(input_path),
        "--out-dir",
        str(scan_dir),
        "--values",
        *[f"{value:.16g}" for value in values],
    ]
    if compute_solution:
        scan_cmd.append("--compute-solution")
    if compute_transport_matrix:
        scan_cmd.append("--compute-transport-matrix")
    if skip_existing:
        scan_cmd.append("--skip-existing")
    if int(jobs) > 1:
        scan_cmd.extend(["--jobs", str(int(jobs))])

    promotion_cmd = [
        sys.executable,
        "examples/optimization/evaluate_sfincs_jax_promotion_scan.py",
        "--scan-dir",
        str(scan_dir),
        "--out-dir",
        str(scan_dir / "promotion_audit"),
        "--stem",
        str(promotion_stem),
        "--target-impurity-flux",
        f"{float(target_impurity_flux):.16g}",
    ]
    if not require_electron_root:
        promotion_cmd.append("--allow-no-electron-root")
    if impurity_species_index is not None:
        promotion_cmd.extend(["--impurity-species-index", str(int(impurity_species_index))])

    return CandidateScanPlan(
        proxy_summary=proxy_path,
        input_namelist=input_path,
        out_dir=scan_dir,
        er_values=values,
        compute_solution=bool(compute_solution),
        compute_transport_matrix=bool(compute_transport_matrix),
        jobs=int(jobs),
        skip_existing=bool(skip_existing),
        scan_command=tuple(scan_cmd),
        promotion_command=tuple(promotion_cmd),
    )


def write_candidate_scan_plan(path: str | Path, plan: CandidateScanPlan, *, proxy_payload: dict[str, Any] | None = None) -> Path:
    """Write a scan plan and selected proxy metadata as JSON."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = plan.as_dict()
    if proxy_payload is not None:
        payload["proxy_workflow"] = proxy_payload.get("workflow")
        payload["proxy_objective_preset"] = proxy_payload.get("objective_preset")
        payload["proxy_final_components"] = proxy_payload.get("final_components")
        payload["proxy_autodiff_gradient_gate"] = proxy_payload.get("autodiff_gradient_gate")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


__all__ = [
    "CandidateScanPlan",
    "build_candidate_scan_plan",
    "er_values_from_bounds",
    "load_proxy_summary",
    "write_candidate_scan_plan",
]
