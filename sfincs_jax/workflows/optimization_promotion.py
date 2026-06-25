"""High-fidelity promotion gates for optimization candidates.

The proxy objectives in :mod:`sfincs_jax.workflows.optimization_objectives` are meant for
fast optimizer steering.  This module evaluates completed ``sfincs_jax`` scan
outputs before a candidate is promoted to a physics claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..ambipolar import radial_current_from_output
from ..io import read_sfincs_h5
from .optimization_objectives import (
    AmbipolarRoot,
    bootstrap_current_objective,
    electron_root_penalty,
    find_ambipolar_roots,
    flux_selectivity_objective,
    kinetic_validation_gate,
)


@dataclass(frozen=True)
class ScanPromotionRun:
    """One completed Er-scan point used in a promotion audit."""

    path: Path
    er: float
    radial_current: float
    bootstrap_current: float
    particle_flux: tuple[float, ...]
    heat_flux: tuple[float, ...]
    residual_norm: float | None
    residual_target: float | None
    residual_gate: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "er": float(self.er),
            "radial_current": float(self.radial_current),
            "bootstrap_current": float(self.bootstrap_current),
            "particle_flux": [float(v) for v in self.particle_flux],
            "heat_flux": [float(v) for v in self.heat_flux],
            "residual_norm": None if self.residual_norm is None else float(self.residual_norm),
            "residual_target": None if self.residual_target is None else float(self.residual_target),
            "residual_gate": self.residual_gate,
        }


@dataclass(frozen=True)
class ScanPromotionSummary:
    """Machine-readable summary for a promoted optimization candidate."""

    scan_dir: Path
    runs: tuple[ScanPromotionRun, ...]
    selected_root: AmbipolarRoot | None
    bootstrap_objective: float
    flux_objective: dict[str, float] | None
    electron_root_penalty: float
    gate_status: str
    failures: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": "sfincs_jax_optimization_high_fidelity_promotion",
            "scan_dir": str(self.scan_dir),
            "runs": [run.as_dict() for run in self.runs],
            "selected_root": None if self.selected_root is None else self.selected_root.as_dict(),
            "bootstrap_objective": float(self.bootstrap_objective),
            "flux_objective": self.flux_objective,
            "electron_root_penalty": float(self.electron_root_penalty),
            "gate_status": self.gate_status,
            "failures": list(self.failures),
            "claim_boundary": (
                "This audit evaluates completed sfincs_jax kinetic outputs. "
                "It promotes a proxy optimization candidate only if the scan is "
                "bracketed, residual-gated, and the requested observables are present."
            ),
        }


def _scalar(data: dict[str, Any], key: str, *, default: float | None = None) -> float | None:
    if key not in data:
        return default
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.size == 0:
        return default
    return float(arr.reshape((-1,))[-1])


def _last_species_values(data: dict[str, Any], key: str, *, n_species: int) -> np.ndarray:
    if key not in data:
        return np.zeros((int(n_species),), dtype=np.float64)
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.ndim == 0:
        return np.full((int(n_species),), float(arr), dtype=np.float64)
    if arr.ndim == 1:
        values = arr.reshape((-1,))
        if values.size == int(n_species):
            return values.astype(np.float64)
        return np.full((int(n_species),), float(values[-1]), dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[0] == int(n_species):
            return arr[:, -1].astype(np.float64)
        if arr.shape[1] == int(n_species):
            return arr[-1, :].astype(np.float64)
    raise ValueError(f"{key} has unsupported shape {arr.shape} for n_species={n_species}")


def _output_paths(scan_dir: Path) -> list[Path]:
    direct = scan_dir / "sfincsOutput.h5"
    if direct.exists():
        return [direct]
    paths = sorted(path for path in scan_dir.glob("*/sfincsOutput.h5") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No sfincsOutput.h5 files found under {scan_dir}")
    return paths


def _read_run(path: Path, *, max_residual_ratio: float) -> ScanPromotionRun:
    data = read_sfincs_h5(path)
    er = _scalar(data, "Er")
    if er is None:
        raise KeyError(f"{path} is missing Er")
    n_species = int(_scalar(data, "Nspecies", default=1.0) or 1)
    radial_current = radial_current_from_output(data)
    bootstrap = _scalar(data, "FSABjHatOverRootFSAB2", default=None)
    if bootstrap is None:
        bootstrap = _scalar(data, "FSABjHat", default=0.0) or 0.0
    particle = _last_species_values(data, "particleFlux_vm_rHat", n_species=n_species)
    heat = _last_species_values(data, "heatFlux_vm_rHat", n_species=n_species)
    residual_norm = _scalar(data, "linearSolverResidualNorm", default=None)
    residual_target = _scalar(data, "linearSolverResidualTarget", default=None)
    residual_gate = kinetic_validation_gate(
        residual_norm=residual_norm,
        residual_target=residual_target,
    )
    if (
        residual_norm is not None
        and residual_target is not None
        and residual_target > 0.0
        and residual_norm <= float(max_residual_ratio) * residual_target
    ):
        residual_gate = {**residual_gate, "status": "pass", "failures": []}
    return ScanPromotionRun(
        path=path,
        er=float(er),
        radial_current=float(radial_current),
        bootstrap_current=float(bootstrap),
        particle_flux=tuple(float(v) for v in particle),
        heat_flux=tuple(float(v) for v in heat),
        residual_norm=residual_norm,
        residual_target=residual_target,
        residual_gate=residual_gate,
    )


def _interp_at_root(xs: np.ndarray, ys: np.ndarray, x0: float) -> np.ndarray:
    if ys.ndim == 1:
        return np.asarray(np.interp(float(x0), xs, ys), dtype=np.float64)
    cols = [np.interp(float(x0), xs, ys[:, i]) for i in range(ys.shape[1])]
    return np.asarray(cols, dtype=np.float64)


def evaluate_sfincs_scan_promotion(
    scan_dir: str | Path,
    *,
    require_electron_root: bool = False,
    impurity_species_index: int | None = None,
    target_impurity_flux: float = 0.0,
    bootstrap_normalizer: float = 1.0,
    max_residual_ratio: float = 1.0,
    require_residuals: bool = True,
) -> ScanPromotionSummary:
    """Evaluate completed SFINCS outputs for optimization promotion.

    Parameters
    ----------
    scan_dir:
        Directory containing either one ``sfincsOutput.h5`` or subdirectories
        from ``sfincs_jax scan-er``.
    require_electron_root:
        If true, fail the promotion gate unless a resolved positive ambipolar
        root is present.
    impurity_species_index:
        Optional species index used for flux selectivity.  If omitted, the flux
        objective is not evaluated.
    target_impurity_flux:
        Outward impurity flux target for the flux-selectivity gate.
    bootstrap_normalizer:
        Scale for the bootstrap-current least-squares objective.
    max_residual_ratio:
        Accept residuals with ``residual_norm <= max_residual_ratio * target``.
    require_residuals:
        If true, missing residual diagnostics fail the promotion gate.
    """

    root_dir = Path(scan_dir).resolve()
    runs = tuple(
        sorted(
            (_read_run(path, max_residual_ratio=max_residual_ratio) for path in _output_paths(root_dir)),
            key=lambda run: run.er,
        )
    )
    er = np.asarray([run.er for run in runs], dtype=np.float64)
    radial_current = np.asarray([run.radial_current for run in runs], dtype=np.float64)
    root_summary = find_ambipolar_roots(er, radial_current)
    selected_root = None
    if require_electron_root and root_summary.electron_roots:
        selected_root = root_summary.electron_roots[0]
    elif root_summary.roots:
        selected_root = root_summary.roots[0]

    electron_penalty = electron_root_penalty(root_summary) if require_electron_root else 0.0
    bootstrap_values = np.asarray([run.bootstrap_current for run in runs], dtype=np.float64)
    bootstrap_objective = bootstrap_current_objective(
        bootstrap_values,
        normalizer=float(bootstrap_normalizer),
    )

    flux_objective = None
    if impurity_species_index is not None and selected_root is not None:
        particle = np.asarray([run.particle_flux for run in runs], dtype=np.float64)
        heat = np.asarray([run.heat_flux for run in runs], dtype=np.float64)
        particle_at_root = _interp_at_root(er, particle, selected_root.er)
        heat_at_root = _interp_at_root(er, heat, selected_root.er)
        flux_objective = flux_selectivity_objective(
            particle_at_root,
            heat_at_root,
            impurity_species_index=int(impurity_species_index),
            target_impurity_flux=float(target_impurity_flux),
        )

    failures: list[str] = []
    if require_electron_root and electron_penalty > 0.0:
        failures.append("required electron root was not found or was not resolved")
    if not root_summary.bracketed:
        failures.append("ambipolar radial-current scan is not bracketed")
    for run in runs:
        missing_residual = run.residual_norm is None or run.residual_target is None
        if require_residuals and missing_residual:
            failures.append(f"{run.path} is missing linear residual diagnostics")
        elif (not missing_residual) and run.residual_gate["status"] != "pass":
            failures.append(f"{run.path} failed residual gate")
    gate_status = "pass" if not failures else "fail"
    return ScanPromotionSummary(
        scan_dir=root_dir,
        runs=runs,
        selected_root=selected_root,
        bootstrap_objective=float(bootstrap_objective),
        flux_objective=flux_objective,
        electron_root_penalty=float(electron_penalty),
        gate_status=gate_status,
        failures=tuple(failures),
    )


__all__ = [
    "ScanPromotionRun",
    "ScanPromotionSummary",
    "evaluate_sfincs_scan_promotion",
]
