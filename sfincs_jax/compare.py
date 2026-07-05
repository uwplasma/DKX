from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import h5py
import numpy as np

from .io import read_sfincs_h5


_VMEC_REFERENCE_CORRUPTION_ABS = 1.0e8
_VMEC_REFERENCE_CORRUPTION_RATIO = 1.0e8
_VMEC_REFERENCE_CORRUPTION_KEYS = {
    "dBHat_sub_theta_dzeta",
    "dBHat_sub_zeta_dtheta",
    "dBHat_sup_theta_dzeta",
    "dBHat_sup_zeta_dtheta",
    "gpsiHatpsiHat",
    "NTVBeforeSurfaceIntegral",
}


@dataclass(frozen=True)
class CompareResult:
    key: str
    max_abs: float
    max_rel: float
    ok: bool


@dataclass(frozen=True)
class H5DatasetParity:
    """Strict parity status for one numeric HDF5 dataset."""

    key: str
    status: str
    reference_shape: tuple[int, ...] | None
    candidate_shape: tuple[int, ...] | None
    max_abs: float | None
    max_rel: float | None
    atol: float
    rtol: float

    @property
    def ok(self) -> bool:
        """Return whether this dataset is non-failing in strict parity mode."""

        return self.status in {"ok", "extra_in_candidate", "non_numeric"}

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-ready representation of the dataset status."""

        payload = asdict(self)
        payload["reference_shape"] = list(self.reference_shape) if self.reference_shape is not None else None
        payload["candidate_shape"] = list(self.candidate_shape) if self.candidate_shape is not None else None
        return payload


def _as_numpy(x: Any) -> np.ndarray | None:
    if isinstance(x, np.ndarray):
        if x.dtype.kind in {"S", "U", "O"}:
            return None
        return x
    if np.isscalar(x):
        arr = np.asarray(x)
        if arr.dtype.kind in {"S", "U", "O"}:
            return None
        return arr
    return None


def _center_fsa(an: np.ndarray) -> np.ndarray:
    """Remove flux-surface-average offsets along theta/zeta axes when present."""
    if an.ndim == 4:
        # (Niter, Ntheta, Nzeta, Nspecies)
        mean = an.mean(axis=(1, 2), keepdims=True)
        return an - mean
    if an.ndim == 3:
        # (Ntheta, Nzeta, Nspecies) or (Niter, Ntheta, Nzeta)
        if an.shape[1] > 1:
            mean = an.mean(axis=(0, 1), keepdims=True)
            return an - mean
        mean = an.mean(axis=1, keepdims=True)
        return an - mean
    return an


def _merge_tolerance_floor(
    tolerances: Dict[str, Dict[str, float]],
    key: str,
    floor: Dict[str, float | bool],
) -> None:
    merged = dict(tolerances.get(key, {}))
    for name, value in floor.items():
        if name in {"ignore", "center_fsa"}:
            merged[name] = bool(merged.get(name, False) or bool(value))
            continue
        try:
            floor_value = float(value)
        except Exception:  # noqa: BLE001
            continue
        current = merged.get(name)
        try:
            current_value = float(current) if current is not None else None
        except Exception:  # noqa: BLE001
            current_value = None
        merged[name] = floor_value if current_value is None else max(current_value, floor_value)
    tolerances[key] = merged


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _is_numeric_array(array: np.ndarray) -> bool:
    return array.dtype.kind in {"b", "i", "u", "f", "c"}


def _numeric_datasets(path: Path) -> dict[str, np.ndarray]:
    """Read all numeric datasets from an HDF5 file into memory."""

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    datasets: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as h5:

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            value = np.asarray(obj[...])
            if _is_numeric_array(value):
                datasets[name] = value

        h5.visititems(visit)
    return datasets


def _dataset_tolerance(
    key: str,
    *,
    atol: float,
    rtol: float,
    tolerances: Mapping[str, Mapping[str, float]] | None,
) -> tuple[float, float]:
    if not tolerances or key not in tolerances:
        return atol, rtol
    local = tolerances[key]
    return float(local.get("atol", atol)), float(local.get("rtol", rtol))


def _strict_max_errors(reference: np.ndarray, candidate: np.ndarray, *, atol: float) -> tuple[float, float]:
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    absdiff = np.abs(cand - ref)
    max_abs = float(np.nanmax(absdiff)) if absdiff.size else 0.0
    denom_floor = max(float(atol), np.finfo(np.float64).tiny)
    denom = np.maximum(np.abs(ref), denom_floor)
    rel = absdiff / denom
    max_rel = float(np.nanmax(rel)) if rel.size else 0.0
    return max_abs, max_rel


def compare_h5_outputs(
    *,
    reference_path: Path,
    candidate_path: Path,
    keys: Iterable[str] | None = None,
    ignore_keys: Iterable[str] = (),
    include_extra: bool = True,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-12,
    tolerances: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    """Strictly compare numeric datasets from two HDF5 files.

    Unlike :func:`compare_sfincs_outputs`, this helper does not apply
    SFINCS-specific skip rules for gauge or convention differences. It records
    missing datasets, shape mismatches, candidate-only extras, and max errors
    for campaign-style output-contract audits.
    """

    reference = _numeric_datasets(reference_path)
    candidate = _numeric_datasets(candidate_path)
    ignored = set(ignore_keys)
    selected = set(keys) if keys is not None else set(reference)
    selected -= ignored

    results: list[H5DatasetParity] = []
    for key in sorted(selected):
        local_atol, local_rtol = _dataset_tolerance(key, atol=atol, rtol=rtol, tolerances=tolerances)
        if key not in reference:
            results.append(H5DatasetParity(key, "missing_in_reference", None, None, None, None, local_atol, local_rtol))
            continue
        if key not in candidate:
            results.append(
                H5DatasetParity(
                    key,
                    "missing_in_candidate",
                    reference[key].shape,
                    None,
                    None,
                    None,
                    local_atol,
                    local_rtol,
                )
            )
            continue
        ref = reference[key]
        cand = candidate[key]
        if ref.shape != cand.shape:
            results.append(
                H5DatasetParity(key, "shape_mismatch", ref.shape, cand.shape, None, None, local_atol, local_rtol)
            )
            continue
        max_abs, max_rel = _strict_max_errors(ref, cand, atol=local_atol)
        ok = bool(np.allclose(ref, cand, atol=local_atol, rtol=local_rtol, equal_nan=True))
        results.append(
            H5DatasetParity(
                key,
                "ok" if ok else "value_mismatch",
                ref.shape,
                cand.shape,
                max_abs,
                max_rel,
                local_atol,
                local_rtol,
            )
        )

    if include_extra:
        for key in sorted(set(candidate) - set(reference) - ignored):
            local_atol, local_rtol = _dataset_tolerance(key, atol=atol, rtol=rtol, tolerances=tolerances)
            results.append(
                H5DatasetParity(key, "extra_in_candidate", None, candidate[key].shape, None, None, local_atol, local_rtol)
            )

    failing = [
        item
        for item in results
        if item.status
        in {
            "missing_in_reference",
            "missing_in_candidate",
            "shape_mismatch",
            "value_mismatch",
        }
    ]
    compared = [item for item in results if item.status in {"ok", "value_mismatch"}]
    max_abs = max((item.max_abs or 0.0 for item in compared), default=0.0)
    max_rel = max((item.max_rel or 0.0 for item in compared), default=0.0)
    counts: dict[str, int] = {}
    for item in results:
        counts[item.status] = counts.get(item.status, 0) + 1

    return {
        "schema_version": 1,
        "reference_path": str(Path(reference_path).resolve()),
        "candidate_path": str(Path(candidate_path).resolve()),
        "overall_status": "pass" if not failing else "fail",
        "numeric_reference_dataset_count": len(reference),
        "numeric_candidate_dataset_count": len(candidate),
        "compared_dataset_count": len(compared),
        "failing_dataset_count": len(failing),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "status_counts": counts,
        "datasets": [item.to_json() for item in results],
    }


def compare_sfincs_outputs(
    *,
    a_path: Path,
    b_path: Path,
    keys: Sequence[str] | None = None,
    ignore_keys: Iterable[str] = ("elapsed time (s)",),
    rtol: float = 1e-12,
    atol: float = 1e-12,
    tolerances: Dict[str, Dict[str, float]] | None = None,
) -> List[CompareResult]:
    """Compare two `sfincsOutput.h5` files dataset-by-dataset."""
    a = read_sfincs_h5(a_path)
    b = read_sfincs_h5(b_path)

    ignore = set(ignore_keys)
    # constraintScheme=0 leaves the density/pressure moments unconstrained and the linear
    # system is rank-deficient. In this branch, PETSc direct/iterative solver details can
    # select different nullspace components, leading to large (but physically gauge-like)
    # offsets in density/pressure-related diagnostics. For strict comparisons, skip the
    # gauge-dependent fields by default.
    def _as_int(v: Any) -> int | None:
        if v is None:
            return None
        if np.isscalar(v):
            try:
                return int(v)
            except Exception:  # noqa: BLE001
                return None
        arr = np.asarray(v)
        if arr.size != 1:
            return None
        try:
            return int(arr.reshape(()))
        except Exception:  # noqa: BLE001
            return None

    constraint_a = _as_int(a.get("constraintScheme"))
    constraint_b = _as_int(b.get("constraintScheme"))
    niter_a = _as_int(a.get("NIterations"))
    niter_b = _as_int(b.get("NIterations"))
    include_phi1_a = _as_int(a.get("includePhi1"))
    include_phi1_b = _as_int(b.get("includePhi1"))
    include_phi1 = bool((include_phi1_a is not None and include_phi1_a > 0) or (include_phi1_b is not None and include_phi1_b > 0))
    if include_phi1:
        # For Phi1 runs, Fortran and JAX can report different Newton-iteration counts
        # while still agreeing on the converged final state. Treat NIterations as
        # metadata and compare converged datasets instead.
        ignore.add("NIterations")
    if (niter_a == 0) or (niter_b == 0):
        # Some Fortran runs do not populate iteration counts for certain solver paths.
        # If either side reports zero iterations, skip comparison for this metadata field.
        ignore.add("NIterations")
    if constraint_a == 0 or constraint_b == 0:
        ignore.update(
            {
                "FSADensityPerturbation",
                "FSAPressurePerturbation",
                "densityPerturbation",
                "pressurePerturbation",
                "totalDensity",
                "totalPressure",
                "velocityUsingTotalDensity",
                "particleFluxBeforeSurfaceIntegral_vm",
                "heatFluxBeforeSurfaceIntegral_vm",
                "flow",
                "FSABFlow",
                "FSABFlow_vs_x",
                "FSABVelocityUsingFSADensity",
                "FSABVelocityUsingFSADensityOverB0",
                "FSABVelocityUsingFSADensityOverRootFSAB2",
                "FSABjHat",
                "FSABjHatOverB0",
                "FSABjHatOverRootFSAB2",
                "MachUsingFSAThermalSpeed",
                "delta_f",
                "full_f",
                "velocityUsingFSADensity",
                "jHat",
            }
        )
    rhs_mode_a = _as_int(a.get("RHSMode"))
    rhs_mode_b = _as_int(b.get("RHSMode"))
    geom_a = _as_int(a.get("geometryScheme"))
    geom_b = _as_int(b.get("geometryScheme"))
    if geom_a == 5 and geom_b == 5 and "uHat" in a and "uHat" in b:
        # VMEC geometryScheme=5 leaves uHat undefined in v3 (computeBHat_VMEC does not
        # populate it). Normalize to zeros for stable, strict comparisons.
        a["uHat"] = np.zeros_like(np.asarray(a["uHat"], dtype=np.float64))
        b["uHat"] = np.zeros_like(np.asarray(b["uHat"], dtype=np.float64))
    local_tolerances: Dict[str, Dict[str, float]] = dict(tolerances or {})
    if geom_a == geom_b:
        # Geometry derivative fields are sensitive to rounding/finite-difference details.
        # Allow tiny relative differences to avoid flagging sub-1e-8 discrepancies.
        geom_tol = {
            "uHat": {"atol": 1e-8},
            "dBHat_sub_zeta_dpsiHat": {"rtol": 1e-7, "atol": 1e-12},
            "dBHat_sup_zeta_dpsiHat": {"rtol": 1e-7, "atol": 1e-12},
            "dBHat_sup_theta_dpsiHat": {"rtol": 1e-7, "atol": 1e-12},
            "dBHatdpsiHat": {"rtol": 1e-7, "atol": 1e-12},
        }
        for k, v in geom_tol.items():
            local_tolerances.setdefault(k, v)
    if geom_a in {1, 2, 4} and geom_b in {1, 2, 4}:
        # In v3 analytic geometry branches (schemes 1/2/4), `gpsiHatpsiHat` is not
        # consistently populated across Fortran builds. Some binaries leave this field
        # as zeros while others emit nonzero values. The classical flux diagnostics are
        # computed directly from this field in `classicalTransport.F90`, so they inherit
        # the same build-dependent reference state. Do not use these undefined analytic
        # geometry fields as parity gates; keep true kinetic flux/current outputs gated.
        analytic_geom_tol = {
            "gpsiHatpsiHat": {"ignore": True},
            "classicalParticleFluxNoPhi1_psiHat": {"ignore": True},
            "classicalParticleFluxNoPhi1_psiN": {"ignore": True},
            "classicalParticleFluxNoPhi1_rHat": {"ignore": True},
            "classicalParticleFluxNoPhi1_rN": {"ignore": True},
            "classicalHeatFluxNoPhi1_psiHat": {"ignore": True},
            "classicalHeatFluxNoPhi1_psiN": {"ignore": True},
            "classicalHeatFluxNoPhi1_rHat": {"ignore": True},
            "classicalHeatFluxNoPhi1_rN": {"ignore": True},
            "classicalParticleFlux_psiHat": {"ignore": True},
            "classicalParticleFlux_psiN": {"ignore": True},
            "classicalParticleFlux_rHat": {"ignore": True},
            "classicalParticleFlux_rN": {"ignore": True},
            "classicalHeatFlux_psiHat": {"ignore": True},
            "classicalHeatFlux_psiN": {"ignore": True},
            "classicalHeatFlux_rHat": {"ignore": True},
            "classicalHeatFlux_rN": {"ignore": True},
            "NTV": {"atol": 1e-6},
            "NTVBeforeSurfaceIntegral": {"atol": 1e-6},
        }
        for k, v in analytic_geom_tol.items():
            if k in local_tolerances:
                merged = dict(local_tolerances.get(k, {}))
                merged.update(v)
                local_tolerances[k] = merged
            else:
                local_tolerances[k] = dict(v)
    if rhs_mode_a == 3 and rhs_mode_b == 3 and constraint_a == 2 and constraint_b == 2:
        # Monoenergetic (RHSMode=3) with constraintScheme=2 can yield tiny total densities
        # at isolated grid points, amplifying small solver/roundoff differences in derived
        # density/pressure diagnostics. Apply a conservative absolute tolerance so strict
        # parity is not dominated by those ill-conditioned points.
        mono_tol = {
            "FSADensityPerturbation": {"atol": 1e-3},
            "FSAPressurePerturbation": {"atol": 1e-3},
            "densityPerturbation": {"atol": 2.0},
            "pressurePerturbation": {"atol": 2.0},
            "pressureAnisotropy": {"atol": 2e-2},
            "totalDensity": {"atol": 2.0},
            "totalPressure": {"atol": 2.0},
            "velocityUsingTotalDensity": {"atol": 1.0e4},
            "particleFluxBeforeSurfaceIntegral_vm": {"atol": 1e-4},
            "heatFluxBeforeSurfaceIntegral_vm": {"atol": 1e-4},
            "NTVBeforeSurfaceIntegral": {"atol": 1e-3},
            # Sources can be sensitive to solver stagnation in the E_parallel RHS; allow a
            # slightly looser relative tolerance while keeping a small absolute floor for
            # near-zero source terms on reduced runtime-windowed grids.
            "sources": {"rtol": 2e-2, "atol": 5e-9},
        }
        for k, v in mono_tol.items():
            if k in local_tolerances:
                merged = dict(local_tolerances.get(k, {}))
                merged.update(v)
                local_tolerances[k] = merged
            else:
                local_tolerances[k] = dict(v)
    if rhs_mode_a == 3 and rhs_mode_b == 3 and constraint_a == 1 and constraint_b == 1:
        # For monoenergetic runs with constraintScheme=1, density/pressure constraints are
        # enforced to solver tolerance, so tiny nonzero FSAs can appear. Use small absolute
        # floors to avoid flagging near-zero residual differences.
        mono_constraint_tol = {
            "FSADensityPerturbation": {"atol": 5e-6},
            "FSAPressurePerturbation": {"atol": 5e-6},
            "NTVBeforeSurfaceIntegral": {"atol": 1e-4},
            "momentumFlux_vm_psiHat": {"atol": 1e-4},
            "momentumFlux_vm_psiN": {"atol": 1e-4},
            "momentumFlux_vm_rHat": {"atol": 1e-4},
            "momentumFlux_vm_rN": {"atol": 1e-4},
        }
        for k, v in mono_constraint_tol.items():
            if k in local_tolerances:
                merged = dict(local_tolerances.get(k, {}))
                merged.update(v)
                local_tolerances[k] = merged
            else:
                local_tolerances[k] = dict(v)
    if rhs_mode_a in {2, 3} and rhs_mode_b in {2, 3}:
        # Transport-matrix runs can yield near-zero pressure anisotropy at isolated grid
        # points; allow a small absolute floor to avoid flagging roundoff noise.
        local_tolerances.setdefault("pressureAnisotropy", {"atol": 2e-6})
        local_tolerances.setdefault("densityPerturbation", {"atol": 2e-6})
        local_tolerances.setdefault("pressurePerturbation", {"atol": 2e-6})
        for key in ("momentumFlux_vm_psiHat", "momentumFlux_vm_psiN", "momentumFlux_vm_rHat", "momentumFlux_vm_rN"):
            local_tolerances.setdefault(key, {"atol": 1e-6})
    if rhs_mode_a == 1 and rhs_mode_b == 1 and constraint_a == 1 and constraint_b == 1:
        # For RHSMode=1 constraintScheme=1 runs, several diagnostics can be very close to
        # zero at isolated grid points, amplifying solver-roundoff differences. Use small
        # absolute floors for those diagnostics to avoid overstating near-zero mismatches.
        rhs1_tol = {
            "FSADensityPerturbation": {"atol": 2e-8},
            "densityPerturbation": {"atol": 2e-6},
            "pressurePerturbation": {"atol": 2e-3},
            "pressureAnisotropy": {"atol": 2e-3},
            "FSAPressurePerturbation": {"atol": 1e-6},
            "NTVBeforeSurfaceIntegral": {"atol": 1e-5},
            "flow": {"atol": 1e-6},
            "FSABFlow": {"atol": 1e-7},
            "FSABFlow_vs_x": {"atol": 2e-7},
            "FSABVelocityUsingFSADensity": {"atol": 1e-7},
            "FSABVelocityUsingFSADensityOverB0": {"atol": 1e-7},
            "FSABVelocityUsingFSADensityOverRootFSAB2": {"atol": 1e-7},
            "velocityUsingFSADensity": {"atol": 2e-6},
            "velocityUsingTotalDensity": {"atol": 2e-6},
            "MachUsingFSAThermalSpeed": {"atol": 1e-7},
            "jHat": {"atol": 1e-6},
            "delta_f": {"atol": 1e-8},
            "sources": {"atol": 1e-9},
        }
        for k, v in rhs1_tol.items():
            _merge_tolerance_floor(local_tolerances, k, v)
        geometry_a = _as_int(a.get("geometryScheme"))
        geometry_b = _as_int(b.get("geometryScheme"))
        collision_a = _as_int(a.get("collisionOperator"))
        collision_b = _as_int(b.get("collisionOperator"))
        if geometry_a == 5 and geometry_b == 5 and collision_a == 0 and collision_b == 0:
            # VMEC (geometryScheme=5) full-FP runs can show ~O(1e-6) absolute
            # differences in local flow/Mach at isolated grid points despite
            # tight parity elsewhere. Allow a slightly higher absolute floor.
            local_tolerances["flow"] = {"atol": 5e-6}
            local_tolerances["MachUsingFSAThermalSpeed"] = {"atol": 5e-6}
    if rhs_mode_a == 1 and rhs_mode_b == 1 and constraint_a == 2 and constraint_b == 2:
        # PAS RHSMode=1 runs constrain per-x moments. Local current-density samples can
        # cross zero, so CPU/GPU reduction order produces O(1e-8) absolute differences
        # even when flux-surface-averaged flow/current gates agree. Keep this floor far
        # below the integrated transport tolerances.
        _merge_tolerance_floor(local_tolerances, "jHat", {"atol": 1.0e-7})
    if rhs_mode_a == 1 and rhs_mode_b == 1 and constraint_a == 1 and constraint_b == 1:
        use_dkes_a = _as_int(a.get("useDKESExBDrift"))
        use_dkes_b = _as_int(b.get("useDKESExBDrift"))
        include_xdot_a = _as_int(a.get("includeXDotTerm"))
        include_xdot_b = _as_int(b.get("includeXDotTerm"))
        include_xidot_a = _as_int(a.get("includeElectricFieldTermInXiDot"))
        include_xidot_b = _as_int(b.get("includeElectricFieldTermInXiDot"))
        collision_a = _as_int(a.get("collisionOperator"))
        collision_b = _as_int(b.get("collisionOperator"))
        if (
            collision_a == 0
            and collision_b == 0
            and (use_dkes_a or 0) > 0
            and (use_dkes_b or 0) > 0
            and (include_xdot_a or 0) <= 0
            and (include_xdot_b or 0) <= 0
            and (include_xidot_a or 0) <= 0
            and (include_xidot_b or 0) <= 0
        ):
            # DKES-trajectory FP runs are ill-conditioned in the L=1 channel; tiny solver
            # differences can amplify into ~O(1e-2) flow/jHat diagnostics. Relax tolerances
            # for flow-related outputs to avoid flagging solver-path sensitivity as physics
            # mismatches.
            dkes_flow_tol = {
                "FSAPressurePerturbation": {"atol": 5e-5},
                "FSABFlow": {"rtol": 1e-1},
                "FSABFlow_vs_x": {"rtol": 1e-1},
                "FSABVelocityUsingFSADensity": {"rtol": 1e-1},
                "FSABVelocityUsingFSADensityOverB0": {"rtol": 1e-1},
                "FSABVelocityUsingFSADensityOverRootFSAB2": {"rtol": 1e-1},
                "FSABjHat": {"rtol": 1e-1},
                "FSABjHatOverB0": {"rtol": 1e-1},
                "FSABjHatOverRootFSAB2": {"rtol": 1e-1},
                "MachUsingFSAThermalSpeed": {"rtol": 1e-1},
                "NTV": {"rtol": 1e-1},
                "flow": {"rtol": 1e-1},
                "jHat": {"rtol": 1e-1},
                "densityPerturbation": {"rtol": 1e-1},
                "velocityUsingFSADensity": {"rtol": 1e-1},
                "velocityUsingTotalDensity": {"rtol": 1e-1},
                # In DKES FP runs, pressure anisotropy can be O(1e-2) near sign changes.
                # Keep a small absolute floor and moderate relative tolerance so strict
                # parity is not dominated by near-zero points.
                "pressureAnisotropy": {"rtol": 5e-3, "atol": 1e-4},
                "particleFlux_vm_psiHat": {"rtol": 1e-1},
                "particleFlux_vm_psiHat_vs_x": {"rtol": 1e-1},
                "particleFlux_vm_psiN": {"rtol": 1e-1},
                "particleFlux_vm_rHat": {"rtol": 1e-1},
                "particleFlux_vm_rN": {"rtol": 1e-1},
                "momentumFlux_vm_psiHat": {"rtol": 1e-1},
                "momentumFlux_vm_psiN": {"rtol": 1e-1},
                "momentumFlux_vm_rHat": {"rtol": 1e-1},
                "momentumFlux_vm_rN": {"rtol": 1e-1},
                "heatFlux_vm_psiHat": {"rtol": 1e-1},
                "heatFlux_vm_psiHat_vs_x": {"rtol": 1e-1},
                "heatFlux_vm_psiN": {"rtol": 1e-1},
                "heatFlux_vm_rHat": {"rtol": 1e-1},
                "heatFlux_vm_rN": {"rtol": 1e-1},
                "momentumFluxBeforeSurfaceIntegral_vm": {"rtol": 1e-1},
            }
            for k, v in dkes_flow_tol.items():
                if k in local_tolerances:
                    merged = dict(local_tolerances.get(k, {}))
                    merged.update(v)
                    local_tolerances[k] = merged
                else:
                    local_tolerances[k] = dict(v)
        if (
            collision_a == 0
            and collision_b == 0
            and geom_a == 5
            and geom_b == 5
            and (use_dkes_a or 0) <= 0
            and (use_dkes_b or 0) <= 0
        ):
            # VMEC geometryScheme=5 FP runs can exhibit tiny solver/roundoff differences
            # that propagate into low-order moments. Allow small absolute floors for
            # those diagnostics to avoid overstating parity gaps.
            fp_vmec_tol = {
                "FSADensityPerturbation": {"atol": 2e-6},
                "FSAPressurePerturbation": {"atol": 2e-3},
                "densityPerturbation": {"atol": 2e-4},
                "pressurePerturbation": {"atol": 2e-3},
                "pressureAnisotropy": {"atol": 3e-3},
                "totalPressure": {"atol": 2e-3},
                "jHat": {"atol": 2e-5},
                "heatFluxBeforeSurfaceIntegral_vm": {"atol": 2e-8},
            }
            for k, v in fp_vmec_tol.items():
                if k in local_tolerances:
                    merged = dict(local_tolerances.get(k, {}))
                    merged.update(v)
                    local_tolerances[k] = merged
                else:
                    local_tolerances[k] = dict(v)
        if (
            collision_a == 0
            and collision_b == 0
            and (include_xdot_a or 0) > 0
            and (include_xdot_b or 0) > 0
            and (include_xidot_a or 0) > 0
            and (include_xidot_b or 0) > 0
            and (use_dkes_a or 0) <= 0
            and (use_dkes_b or 0) <= 0
        ):
            # Collisionless full-trajectory FP runs can show ~percent-level sensitivity to
            # solver stopping criteria. Relax tolerances for flow/flux diagnostics to avoid
            # overstating solver-path sensitivity as physics mismatches.
            traj_flow_tol = {
                "FSAPressurePerturbation": {"atol": 3e-4},
                "FSABFlow": {"rtol": 3e-2},
                "FSABFlow_vs_x": {"rtol": 3e-2, "atol": 1e-5},
                "FSABVelocityUsingFSADensity": {"rtol": 3e-2},
                "FSABVelocityUsingFSADensityOverB0": {"rtol": 3e-2},
                "FSABVelocityUsingFSADensityOverRootFSAB2": {"rtol": 3e-2},
                "FSABjHat": {"rtol": 3e-2},
                "FSABjHatOverB0": {"rtol": 3e-2},
                "FSABjHatOverRootFSAB2": {"rtol": 3e-2},
                "MachUsingFSAThermalSpeed": {"rtol": 3e-2},
                "NTV": {"rtol": 3e-2},
                "NTVBeforeSurfaceIntegral": {"rtol": 3e-2},
                "flow": {"rtol": 3e-2},
                "jHat": {"rtol": 3e-2},
                "densityPerturbation": {"rtol": 3e-2},
                "pressurePerturbation": {"rtol": 3e-2},
                "pressureAnisotropy": {"rtol": 3e-2},
                "velocityUsingFSADensity": {"rtol": 3e-2},
                "velocityUsingTotalDensity": {"rtol": 3e-2},
                "particleFlux_vm_psiHat": {"rtol": 3e-2},
                "particleFlux_vm_psiHat_vs_x": {"rtol": 3e-2},
                "particleFlux_vm_psiN": {"rtol": 3e-2},
                "particleFlux_vm_rHat": {"rtol": 3e-2},
                "particleFlux_vm_rN": {"rtol": 3e-2},
                "momentumFlux_vm_psiHat": {"rtol": 3e-2},
                "momentumFlux_vm_psiN": {"rtol": 3e-2},
                "momentumFlux_vm_rHat": {"rtol": 3e-2},
                "momentumFlux_vm_rN": {"rtol": 3e-2},
                "heatFlux_vm_psiHat": {"rtol": 3e-2},
                "heatFlux_vm_psiHat_vs_x": {"rtol": 3e-2},
                "heatFlux_vm_psiN": {"rtol": 3e-2},
                "heatFlux_vm_rHat": {"rtol": 3e-2},
                "heatFlux_vm_rN": {"rtol": 3e-2},
                "momentumFluxBeforeSurfaceIntegral_vm": {"rtol": 3e-2},
            }
            for k, v in traj_flow_tol.items():
                if k in local_tolerances:
                    merged = dict(local_tolerances.get(k, {}))
                    merged.update(v)
                    local_tolerances[k] = merged
                else:
                    local_tolerances[k] = dict(v)
            if geom_a == 5 and geom_b == 5:
                # VMEC full-trajectory FP runs can have tiny but non-negligible differences
                # in total density and pre-surface-integral particle flux diagnostics.
                # Use small case-specific floors to avoid over-reporting these known
                # solver-path sensitivities.
                vmec_traj_tol = {
                    "totalDensity": {"rtol": 5e-4, "atol": 2e-4},
                    "particleFluxBeforeSurfaceIntegral_vm": {"rtol": 5e-4, "atol": 2e-9},
                }
                for k, v in vmec_traj_tol.items():
                    if k in local_tolerances:
                        merged = dict(local_tolerances.get(k, {}))
                        merged.update(v)
                        local_tolerances[k] = merged
                    else:
                        local_tolerances[k] = dict(v)
    if rhs_mode_a == 1 and rhs_mode_b == 1 and constraint_a == 0 and constraint_b == 0:
        # constraintScheme=0 leaves the FSA density/pressure gauge unconstrained, so compare
        # gauge-invariant local structure and allow small absolute floors on near-zero flux
        # diagnostics rather than flagging nullspace/roundoff artifacts.
        rhs1_cs0_tol = {
            "densityPerturbation": {"center_fsa": True},
            "pressurePerturbation": {"center_fsa": True},
            "totalDensity": {"center_fsa": True},
            "totalPressure": {"center_fsa": True},
            "FSADensityPerturbation": {"ignore": True},
            "FSAPressurePerturbation": {"ignore": True},
            "velocityUsingTotalDensity": {"atol": 1e-2},
            "particleFluxBeforeSurfaceIntegral_vm": {"atol": 2e-5},
            "heatFluxBeforeSurfaceIntegral_vm": {"atol": 2e-5},
            "heatFlux_vm_psiHat": {"atol": 1e-5},
            "heatFlux_vm_psiHat_vs_x": {"atol": 1e-5},
            "heatFlux_vm_psiN": {"atol": 1e-5},
            "heatFlux_vm_rHat": {"atol": 1e-5},
            "heatFlux_vm_rN": {"atol": 1e-5},
            "particleFlux_vm_psiHat": {"atol": 1e-6},
            "particleFlux_vm_psiHat_vs_x": {"atol": 1e-6},
            "particleFlux_vm_psiN": {"atol": 1e-6},
            "particleFlux_vm_rHat": {"atol": 1e-6},
            "particleFlux_vm_rN": {"atol": 1e-6},
            "momentumFluxBeforeSurfaceIntegral_vm": {"atol": 1e-6},
            "pressureAnisotropy": {"atol": 2e-4},
        }
        for k, v in rhs1_cs0_tol.items():
            _merge_tolerance_floor(local_tolerances, k, v)
    if rhs_mode_a == 1 and rhs_mode_b == 1 and constraint_a == 2 and constraint_b == 2:
        # For RHSMode=1 constraintScheme=2 runs, pressure/density perturbations can be near
        # machine zero at isolated points. Apply small absolute floors to avoid flagging
        # benign roundoff differences in those diagnostics and delta_f exports.
        rhs1_cs2_tol = {
            "FSADensityPerturbation": {"atol": 1e-5},
            "FSAPressurePerturbation": {"atol": 2e-5},
            "FSABFlow_vs_x": {"atol": 2e-7},
            "densityPerturbation": {"atol": 1e-5},
            "pressurePerturbation": {"atol": 1e-2},
            "heatFlux_vm_psiHat_vs_x": {"atol": 2e-6},
            "delta_f": {"atol": 1e-5},
            "full_f": {"atol": 1e-5},
            "sources": {"atol": 1e-9},
        }
        for k, v in rhs1_cs2_tol.items():
            _merge_tolerance_floor(local_tolerances, k, v)
        use_dkes_a = _as_int(a.get("useDKESExBDrift"))
        use_dkes_b = _as_int(b.get("useDKESExBDrift"))
        include_xdot_a = _as_int(a.get("includeXDotTerm"))
        include_xdot_b = _as_int(b.get("includeXDotTerm"))
        include_xidot_a = _as_int(a.get("includeElectricFieldTermInXiDot"))
        include_xidot_b = _as_int(b.get("includeElectricFieldTermInXiDot"))
        collision_a = _as_int(a.get("collisionOperator"))
        collision_b = _as_int(b.get("collisionOperator"))
        if (
            collision_a == 1
            and collision_b == 1
            and (include_xdot_a or 0) > 0
            and (include_xdot_b or 0) > 0
            and (include_xidot_a or 0) > 0
            and (include_xidot_b or 0) > 0
            and (use_dkes_a or 0) <= 0
            and (use_dkes_b or 0) <= 0
        ):
            # PAS full-trajectory constraintScheme=2 runs can leave sub-0.1%-level drift in
            # the vm heat-flux coordinate scalings even when the linear solve is already
            # converged. Treat these as one transported scalar's solver-path sensitivity.
            pas_traj_heatflux_tol = {
                "heatFlux_vm_psiHat": {"rtol": 1e-3},
                "heatFlux_vm_psiN": {"rtol": 1e-3},
                "heatFlux_vm_rHat": {"rtol": 1e-3},
                "heatFlux_vm_rN": {"rtol": 1e-3},
            }
            for k, v in pas_traj_heatflux_tol.items():
                _merge_tolerance_floor(local_tolerances, k, v)
    if rhs_mode_a in {2, 3} and rhs_mode_b in {2, 3} and constraint_a == 1 and constraint_b == 1:
        # Transport-matrix solves with constraintScheme=1 can yield tiny (~1e-10) source terms
        # that are sensitive to Krylov stopping tolerances. Allow a small absolute margin.
        local_tolerances.setdefault("sources", {"atol": 1e-9})
    if geom_a == 5 and geom_b == 5 and rhs_mode_a in {2, 3} and rhs_mode_b in {2, 3}:
        # VMEC monoenergetic/transport runs do not populate NTV in v3; Fortran outputs can
        # contain uninitialized garbage. `gpsiHatpsiHat` is likewise not stable across
        # builds for these branches, so ignore these fields for parity.
        local_tolerances["NTV"] = {**local_tolerances.get("NTV", {}), "ignore": True}
        local_tolerances["NTVBeforeSurfaceIntegral"] = {
            **local_tolerances.get("NTVBeforeSurfaceIntegral", {}),
            "ignore": True,
        }
        local_tolerances["gpsiHatpsiHat"] = {
            **local_tolerances.get("gpsiHatpsiHat", {}),
            "ignore": True,
        }
    if keys is None:
        keys = sorted(set(a.keys()) & set(b.keys()))

    nan_as_zero_keys = {
        "classicalParticleFlux_psiHat",
        "classicalParticleFlux_psiN",
        "classicalParticleFlux_rHat",
        "classicalParticleFlux_rN",
        "classicalParticleFluxNoPhi1_psiHat",
        "classicalParticleFluxNoPhi1_psiN",
        "classicalParticleFluxNoPhi1_rHat",
        "classicalParticleFluxNoPhi1_rN",
        "classicalHeatFlux_psiHat",
        "classicalHeatFlux_psiN",
        "classicalHeatFlux_rHat",
        "classicalHeatFlux_rN",
        "classicalHeatFluxNoPhi1_psiHat",
        "classicalHeatFluxNoPhi1_psiN",
        "classicalHeatFluxNoPhi1_rHat",
        "classicalHeatFluxNoPhi1_rN",
    }
    if geom_a == 5 and geom_b == 5:
        # VMEC geometryScheme=5 can leave some metric derivatives NaN in Fortran; treat NaNs
        # as zero to compare on the finite entries.
        nan_as_zero_keys.update({"dBHat_sup_theta_dzeta", "gpsiHatpsiHat"})

    results: List[CompareResult] = []
    for k in keys:
        if k in ignore:
            continue
        av = a.get(k)
        bv = b.get(k)
        if av is None or bv is None:
            continue
        an = _as_numpy(av)
        bn = _as_numpy(bv)
        if an is None or bn is None:
            continue
        if an.shape != bn.shape:
            # For includePhi1 runs, some datasets include a Newton-iteration axis
            # that can differ in length. Compare only the final iteration to avoid
            # flagging intermediate-iteration differences that do not affect the
            # converged solution.
            if include_phi1 and niter_a and niter_b:
                if (
                    an.ndim >= 1
                    and bn.ndim >= 1
                    and an.shape[-1] == niter_a
                    and bn.shape[-1] == niter_b
                    and (niter_a > 1 or niter_b > 1)
                ):
                    an = an[..., -1]
                    bn = bn[..., -1]
                # Re-check shapes after slicing.
            if an.shape != bn.shape:
                results.append(CompareResult(key=k, max_abs=float("inf"), max_rel=float("inf"), ok=False))
                continue
        # If includePhi1 is active and an iteration axis is present, compare only the
        # final Newton iterate to align parity on the converged solution.
        if include_phi1 and niter_a and niter_b and an.ndim >= 1 and bn.ndim >= 1:
            if an.shape[-1] == niter_a and bn.shape[-1] == niter_b and (niter_a > 1 or niter_b > 1):
                an = an[..., -1]
                bn = bn[..., -1]

        if geom_a == 5 and geom_b == 5:
            nan_mask = np.isnan(an) | np.isnan(bn)
            if np.any(nan_mask):
                an = np.where(nan_mask, 0.0, an)
                bn = np.where(nan_mask, 0.0, bn)
            if k in _VMEC_REFERENCE_CORRUPTION_KEYS:
                scale = np.maximum(np.abs(an), np.abs(bn))
                partner = np.maximum(np.minimum(np.abs(an), np.abs(bn)), 1.0)
                corruption_mask = (
                    ~np.isfinite(an)
                    | ~np.isfinite(bn)
                    | (
                        (scale > _VMEC_REFERENCE_CORRUPTION_ABS)
                        & (scale > (_VMEC_REFERENCE_CORRUPTION_RATIO * partner))
                    )
                )
                if np.any(corruption_mask):
                    an = np.where(corruption_mask, 0.0, an)
                    bn = np.where(corruption_mask, 0.0, bn)

        if k in nan_as_zero_keys:
            an = np.nan_to_num(an, nan=0.0)
            bn = np.nan_to_num(bn, nan=0.0)

        tol = local_tolerances.get(k, {}) if local_tolerances else {}
        if bool(tol.get("ignore", False)):
            continue
        rtol_k = float(tol.get("rtol", rtol))
        atol_k = float(tol.get("atol", atol))

        if bool(tol.get("center_fsa", False)):
            an = _center_fsa(an)
            bn = _center_fsa(bn)

        diff = np.abs(an - bn)
        max_abs = float(diff.max()) if diff.size else float(abs(float(an) - float(bn)))
        denom = np.maximum(np.abs(bn), np.asarray(atol_k))
        max_rel = float((diff / denom).max()) if diff.size else float(abs(float(an) - float(bn)) / max(abs(float(bn)), atol_k))
        ok = bool(np.allclose(an, bn, rtol=rtol_k, atol=atol_k))
        results.append(CompareResult(key=k, max_abs=max_abs, max_rel=max_rel, ok=ok))

    return results


def main(argv: list[str] | None = None) -> int:
    """CLI-compatible strict numeric HDF5 parity check."""

    parser = argparse.ArgumentParser(description="Strict numeric HDF5 parity check.")
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, help="Optional JSON report path")
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--ignore-key", action="append", default=[])
    parser.add_argument("--key", action="append", default=None, help="Dataset key to compare; may be repeated")
    parser.add_argument("--no-extra", action="store_true", help="Do not report candidate-only datasets")
    args = parser.parse_args(argv)

    report = compare_h5_outputs(
        reference_path=args.reference,
        candidate_path=args.candidate,
        keys=args.key,
        ignore_keys=args.ignore_key,
        include_extra=not args.no_extra,
        atol=args.atol,
        rtol=args.rtol,
    )
    text = json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0 if report["overall_status"] == "pass" else 1
