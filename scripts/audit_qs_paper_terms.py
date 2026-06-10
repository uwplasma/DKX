#!/usr/bin/env python
"""Term-by-term SFINCS Fortran v3 / SFINCS_JAX audit for the QS-paper decks.

The default mode is intentionally cheap: it regenerates SFINCS_JAX geometry-only
outputs at the archived Fortran angular grid and compares geometry
interpolation, radial-gradient conversion, finite-beta switches, and any
available current-moment assembly fields.  Use ``--compute-solution`` only when
a solved JAX RHSMode=1 comparison is needed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_ENABLE_X64", "1")

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.io import write_sfincs_jax_output_h5  # noqa: E402


DEFAULT_ZENODO_ROOT = Path("/Users/rogeriojorge/local/20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current")
BENCHMARK_REL = Path("calculations/20211226-01-sfincs_for_precise_QS_for_Redl_benchmark")


@dataclass(frozen=True)
class BenchmarkCase:
    key: str
    case_dir: str
    wout_name: str


CASES = {
    "QA": BenchmarkCase(
        key="QA",
        case_dir="20211226-01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces",
        wout_name="wout_new_QA_aScaling.nc",
    ),
    "QH": BenchmarkCase(
        key="QH",
        case_dir="20211226-01-019_QH_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces",
        wout_name="wout_new_QH_aScaling.nc",
    ),
}


GEOMETRY_SCALARS = (
    "psiN",
    "rN",
    "psiHat",
    "psiAHat",
    "aHat",
    "iota",
    "diotadpsiHat",
    "GHat",
    "IHat",
    "VPrimeHat",
    "FSABHat2",
    "B0OverBBar",
    "NPeriods",
    "geometryScheme",
)

GEOMETRY_ARRAYS = (
    "theta",
    "zeta",
    "BHat",
    "DHat",
    "dBHatdtheta",
    "dBHatdzeta",
    "dBHatdpsiHat",
    "BHat_sub_theta",
    "BHat_sub_zeta",
    "BHat_sub_psi",
    "BHat_sup_theta",
    "BHat_sup_zeta",
    "dBHat_sub_theta_dpsiHat",
    "dBHat_sub_zeta_dpsiHat",
    "dBHat_sub_theta_dzeta",
    "dBHat_sub_zeta_dtheta",
    "dBHat_sub_psi_dtheta",
    "dBHat_sub_psi_dzeta",
    "gpsiHatpsiHat",
    "BDotCurlB",
)

GRADIENT_FIELDS = (
    "inputRadialCoordinate",
    "inputRadialCoordinateForGradients",
    "dPhiHatdpsiHat",
    "dPhiHatdpsiN",
    "dPhiHatdrHat",
    "dPhiHatdrN",
    "Er",
    "dnHatdpsiHat",
    "dnHatdpsiN",
    "dnHatdrHat",
    "dnHatdrN",
    "dTHatdpsiHat",
    "dTHatdpsiN",
    "dTHatdrHat",
    "dTHatdrN",
)

FINITE_BETA_FLAGS = (
    "includeXDotTerm",
    "includeElectricFieldTermInXiDot",
    "useDKESExBDrift",
    "includePhi1",
    "includePhi1InKineticEquation",
    "includePhi1InCollisionOperator",
    "force0RadialCurrentInEquilibrium",
    "magneticDriftScheme",
    "magneticDriftDerivativeScheme",
    "collisionOperator",
    "constraintScheme",
)

CURRENT_FIELDS = (
    "Zs",
    "FSABFlow",
    "FSABFlow_vs_x",
    "FSABjHat",
    "FSABjHatOverB0",
    "FSABjHatOverRootFSAB2",
)


FORTRAN_SOURCE_MAP = {
    "vmec_geometry": "/Users/rogeriojorge/local/sfincs/fortran/version3/geometry.F90:2287-2931",
    "radial_gradients": "/Users/rogeriojorge/local/sfincs/fortran/version3/radialCoordinates.F90:171-240",
    "current_assembly": "/Users/rogeriojorge/local/sfincs/fortran/version3/diagnostics.F90:390-746",
    "hdf5_output": "/Users/rogeriojorge/local/sfincs/fortran/version3/writeHDF5Output.F90:199-603",
}


def _surface_dir_name(s_value: float) -> str:
    return f"psiN_{float(s_value):.12g}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_array(h5: h5py.File, key: str) -> np.ndarray | None:
    if key not in h5:
        return None
    return np.asarray(h5[key])


def _stats(arr: np.ndarray) -> dict[str, Any]:
    a = np.asarray(arr)
    if a.size == 0:
        return {"shape": list(a.shape), "size": 0}
    if not np.issubdtype(a.dtype, np.number):
        return {"shape": list(a.shape), "dtype": str(a.dtype), "size": int(a.size)}
    af = np.asarray(a, dtype=np.float64)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "size": int(a.size),
        "min": float(np.nanmin(af)),
        "max": float(np.nanmax(af)),
        "mean": float(np.nanmean(af)),
        "rms": float(np.sqrt(np.nanmean(af * af))),
    }


def relative_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    layout = "direct"
    if ref.shape != cand.shape:
        if ref.T.shape == cand.shape:
            ref = ref.T
            layout = "fortran_transpose"
        elif ref.shape == cand.T.shape:
            cand = cand.T
            layout = "jax_transpose"
        else:
            return {
                "status": "shape_mismatch",
                "fortran_shape": list(np.asarray(reference).shape),
                "jax_shape": list(np.asarray(candidate).shape),
            }
    diff = cand - ref
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0
    denom = max(
        float(np.max(np.abs(ref))) if ref.size else 0.0,
        float(np.max(np.abs(cand))) if cand.size else 0.0,
        np.finfo(float).tiny,
    )
    return {
        "status": "ok",
        "layout": layout,
        "shape": list(ref.shape),
        "max_abs": max_abs,
        "rms_abs": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
        "max_rel": float(max_abs / denom),
        "fortran_stats": _stats(ref),
        "jax_stats": _stats(cand),
    }


def compare_fields(fortran_path: Path, jax_path: Path, keys: tuple[str, ...]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    with h5py.File(fortran_path, "r") as f_fortran, h5py.File(jax_path, "r") as f_jax:
        for key in keys:
            f_arr = _read_array(f_fortran, key)
            j_arr = _read_array(f_jax, key)
            if f_arr is None and j_arr is None:
                continue
            if f_arr is None:
                report[key] = {"status": "missing_fortran", "jax_stats": _stats(j_arr)}
            elif j_arr is None:
                report[key] = {"status": "missing_jax", "fortran_stats": _stats(f_arr)}
            else:
                report[key] = relative_metrics(f_arr, j_arr)
    return report


def max_ok_relative_error(report: dict[str, Any], keys: tuple[str, ...] | None = None) -> float:
    selected = report if keys is None else {key: report.get(key, {}) for key in keys}
    return max(
        (
            float(item.get("max_rel", 0.0))
            for item in selected.values()
            if isinstance(item, dict) and item.get("status") == "ok"
        ),
        default=0.0,
    )


def _scalar(h5: h5py.File, key: str) -> float | None:
    arr = _read_array(h5, key)
    if arr is None:
        return None
    return float(np.asarray(arr, dtype=np.float64).reshape(-1)[0])


def _conversion_factors(h5: h5py.File) -> dict[str, float] | None:
    psi_a_hat = _scalar(h5, "psiAHat")
    a_hat = _scalar(h5, "aHat")
    r_n = _scalar(h5, "rN")
    if psi_a_hat is None or a_hat is None or r_n is None:
        return None
    return {
        "ddpsiN2ddpsiHat": 1.0 / psi_a_hat,
        "ddpsiHat2ddpsiN": psi_a_hat,
        "ddrHat2ddpsiHat": a_hat / (2.0 * psi_a_hat * r_n),
        "ddpsiHat2ddrHat": 2.0 * psi_a_hat * r_n / a_hat,
        "ddrN2ddpsiHat": 1.0 / (2.0 * psi_a_hat * r_n),
        "ddpsiHat2ddrN": 2.0 * psi_a_hat * r_n,
    }


def _check_conversion(h5: h5py.File, *, base: str, source: str, factor: float, target: str) -> dict[str, Any] | None:
    src = _read_array(h5, source)
    tgt = _read_array(h5, target)
    if src is None or tgt is None:
        return None
    expected = float(factor) * np.asarray(src, dtype=np.float64)
    metrics = relative_metrics(expected, np.asarray(tgt, dtype=np.float64))
    metrics["expected"] = f"{target} = {factor:.16e} * {source}"
    metrics["base"] = base
    return metrics


def gradient_conversion_audit(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        conv = _conversion_factors(h5)
        if conv is None:
            return {"status": "missing_coordinate_scalars"}
        grad_coord = int(_scalar(h5, "inputRadialCoordinateForGradients") or -999)
        checks: dict[str, Any] = {"status": "ok", "inputRadialCoordinateForGradients": grad_coord, "factors": conv}
        if grad_coord == 1:
            density_source, temp_source, phi_source = "dnHatdpsiN", "dTHatdpsiN", "dPhiHatdpsiN"
            factor_species = conv["ddpsiN2ddpsiHat"]
            factor_phi = conv["ddpsiN2ddpsiHat"]
        elif grad_coord in {2, 4}:
            density_source, temp_source, phi_source = "dnHatdrHat", "dTHatdrHat", "Er"
            factor_species = conv["ddrHat2ddpsiHat"]
            factor_phi = -conv["ddrHat2ddpsiHat"]
        elif grad_coord == 3:
            density_source, temp_source, phi_source = "dnHatdrN", "dTHatdrN", "dPhiHatdrN"
            factor_species = conv["ddrN2ddpsiHat"]
            factor_phi = conv["ddrN2ddpsiHat"]
        elif grad_coord == 0:
            density_source, temp_source, phi_source = "dnHatdpsiHat", "dTHatdpsiHat", "dPhiHatdpsiHat"
            factor_species = 1.0
            factor_phi = 1.0
        else:
            return {**checks, "status": "unsupported_gradient_coordinate"}

        for label, source, factor, target in (
            ("density", density_source, factor_species, "dnHatdpsiHat"),
            ("temperature", temp_source, factor_species, "dTHatdpsiHat"),
            ("potential", phi_source, factor_phi, "dPhiHatdpsiHat"),
        ):
            item = _check_conversion(h5, base=label, source=source, factor=factor, target=target)
            if item is not None:
                checks[label] = item
        for source, factor, target in (
            ("dnHatdpsiHat", conv["ddpsiHat2ddpsiN"], "dnHatdpsiN"),
            ("dTHatdpsiHat", conv["ddpsiHat2ddpsiN"], "dTHatdpsiN"),
            ("dPhiHatdpsiHat", conv["ddpsiHat2ddpsiN"], "dPhiHatdpsiN"),
            ("dnHatdpsiHat", conv["ddpsiHat2ddrHat"], "dnHatdrHat"),
            ("dTHatdpsiHat", conv["ddpsiHat2ddrHat"], "dTHatdrHat"),
            ("dPhiHatdpsiHat", conv["ddpsiHat2ddrHat"], "dPhiHatdrHat"),
            ("dnHatdpsiHat", conv["ddpsiHat2ddrN"], "dnHatdrN"),
            ("dTHatdpsiHat", conv["ddpsiHat2ddrN"], "dTHatdrN"),
            ("dPhiHatdpsiHat", conv["ddpsiHat2ddrN"], "dPhiHatdrN"),
        ):
            item = _check_conversion(h5, base="derived", source=source, factor=factor, target=target)
            if item is not None:
                checks[f"{source}_to_{target}"] = item
        return checks


def fsabjhat_assembly_audit(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        z_s = _read_array(h5, "Zs")
        flow = _read_array(h5, "FSABFlow")
        current = _read_array(h5, "FSABjHat")
        if z_s is None or flow is None or current is None:
            return {"status": "missing_current_fields"}
        z = np.asarray(z_s, dtype=np.float64).reshape(-1)
        flow_arr = np.asarray(flow, dtype=np.float64)
        if flow_arr.shape[0] == z.size:
            assembled = np.einsum("s,s...->...", z, flow_arr)
        elif flow_arr.shape[-1] == z.size:
            assembled = np.einsum("s,...s->...", z, flow_arr)
        else:
            return {"status": "shape_mismatch", "Zs_shape": list(z.shape), "FSABFlow_shape": list(flow_arr.shape)}

        report = {
            "status": "ok",
            "FSABjHat_from_Zs_dot_FSABFlow": relative_metrics(np.asarray(current, dtype=np.float64).reshape(assembled.shape), assembled),
        }
        b0 = _scalar(h5, "B0OverBBar")
        fsab2 = _scalar(h5, "FSABHat2")
        if b0 is not None and "FSABjHatOverB0" in h5:
            report["FSABjHatOverB0"] = relative_metrics(np.asarray(h5["FSABjHatOverB0"]), assembled / b0)
        if fsab2 is not None and "FSABjHatOverRootFSAB2" in h5:
            report["FSABjHatOverRootFSAB2"] = relative_metrics(np.asarray(h5["FSABjHatOverRootFSAB2"]), assembled / np.sqrt(fsab2))
        flow_vs_x = _read_array(h5, "FSABFlow_vs_x")
        if flow_vs_x is not None:
            summed = np.sum(np.asarray(flow_vs_x, dtype=np.float64), axis=0)
            report["FSABFlow_vs_x_sum"] = relative_metrics(flow_arr, summed)
        return report


def finite_beta_flags(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        return {key: _jsonable(np.asarray(h5[key]).reshape(-1)[0]) for key in FINITE_BETA_FLAGS if key in h5}


def _write_jax_output(
    *,
    fortran_input: Path,
    wout_path: Path,
    output_path: Path,
    force: bool,
    compute_solution: bool,
    solve_method: str | None,
    verbose: bool,
) -> Path:
    if output_path.exists() and not force:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_sfincs_jax_output_h5(
        input_namelist=fortran_input,
        output_path=output_path,
        wout_path=wout_path,
        overwrite=True,
        compute_solution=compute_solution,
        compute_transport_matrix=False,
        solve_method=solve_method,
        verbose=verbose,
    )
    return output_path


def _case_paths(args: argparse.Namespace) -> tuple[BenchmarkCase, Path, Path]:
    case = CASES[str(args.case).upper()]
    benchmark_root = args.zenodo_root.expanduser().resolve() / BENCHMARK_REL
    return case, benchmark_root / case.case_dir, benchmark_root / case.wout_name


def _replace_namelist_scalar(text: str, key: str, value: int | float) -> str:
    pattern = re.compile(rf"(?im)^(\s*{re.escape(key)}\s*=\s*)([^\s!]+)(.*)$")
    replacement = rf"\g<1>{value}\g<3>"
    new_text, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        raise ValueError(f"Could not find namelist scalar {key!r} to patch.")
    return new_text


def _patched_input_path(args: argparse.Namespace, *, source: Path, surface: str) -> Path:
    overrides = {
        "Ntheta": args.ntheta,
        "Nzeta": args.nzeta,
        "Nxi": args.nxi,
        "Nx": args.nx,
        "solverTolerance": args.solver_tolerance,
    }
    active = {key: value for key, value in overrides.items() if value is not None}
    if not active:
        return source
    patched = args.run_root / "patched_inputs" / args.case / surface / "input.namelist"
    patched.parent.mkdir(parents=True, exist_ok=True)
    text = source.read_text(encoding="utf-8")
    for key, value in active.items():
        text = _replace_namelist_scalar(text, key, value)
    patched.write_text(text, encoding="utf-8")
    return patched


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="QH")
    parser.add_argument("--s-values", default="0.5,0.7", help="Comma-separated psiN surfaces.")
    parser.add_argument("--zenodo-root", type=Path, default=Path(os.environ.get("SFINCS_JAX_QS_ZENODO_ROOT", DEFAULT_ZENODO_ROOT)))
    parser.add_argument("--run-root", type=Path, default=Path("/tmp/sfincs_jax_qs_term_audit"))
    parser.add_argument("--out", type=Path, default=Path("outputs/qs_paper_term_audit.json"))
    parser.add_argument("--compute-solution", action="store_true", help="Run full SFINCS_JAX solves before auditing current assembly.")
    parser.add_argument("--solve-method", default=None)
    parser.add_argument("--ntheta", type=int, default=None, help="Optional SFINCS_JAX Ntheta override for bounded solved audits.")
    parser.add_argument("--nzeta", type=int, default=None, help="Optional SFINCS_JAX Nzeta override for bounded solved audits.")
    parser.add_argument("--nxi", type=int, default=None, help="Optional SFINCS_JAX Nxi override for bounded solved audits.")
    parser.add_argument("--nx", type=int, default=None, help="Optional SFINCS_JAX Nx override for bounded solved audits.")
    parser.add_argument("--solver-tolerance", type=float, default=None, help="Optional SFINCS_JAX solverTolerance override.")
    parser.add_argument("--verbose", action="store_true", help="Emit SFINCS_JAX phase and solver progress during output generation.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    return parser


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    case, case_root, wout_path = _case_paths(args)
    s_values = [float(piece.strip()) for piece in str(args.s_values).split(",") if piece.strip()]
    report: dict[str, Any] = {
        "schema_version": 1,
        "case": case.key,
        "case_root": str(case_root),
        "wout_path": str(wout_path),
        "compute_solution": bool(args.compute_solution),
        "fortran_source_map": FORTRAN_SOURCE_MAP,
        "surfaces": {},
    }
    for s_value in s_values:
        surface = _surface_dir_name(s_value)
        fortran_input = case_root / surface / "input.namelist"
        fortran_output = case_root / surface / "sfincsOutput.h5"
        if not fortran_output.exists():
            raise FileNotFoundError(f"Missing archived Fortran output: {fortran_output}")
        jax_input = _patched_input_path(args, source=fortran_input, surface=surface)
        jax_output = args.run_root / case.key / surface / ("sfincsOutput_jax_solved.h5" if args.compute_solution else "sfincsOutput_jax_geometry.h5")
        _write_jax_output(
            fortran_input=jax_input,
            wout_path=wout_path,
            output_path=jax_output,
            force=bool(args.force),
            compute_solution=bool(args.compute_solution),
            solve_method=args.solve_method,
            verbose=bool(args.verbose),
        )
        surface_report = {
            "fortran_input": str(fortran_input),
            "fortran_output": str(fortran_output),
            "jax_input": str(jax_input),
            "jax_output": str(jax_output),
            "jax_resolution_overrides": {
                "Ntheta": args.ntheta,
                "Nzeta": args.nzeta,
                "Nxi": args.nxi,
                "Nx": args.nx,
                "solverTolerance": args.solver_tolerance,
            },
            "geometry_scalars": compare_fields(fortran_output, jax_output, GEOMETRY_SCALARS),
            "geometry_arrays": compare_fields(fortran_output, jax_output, GEOMETRY_ARRAYS),
            "gradient_fields": compare_fields(fortran_output, jax_output, GRADIENT_FIELDS),
            "finite_beta_flags": {
                "fortran": finite_beta_flags(fortran_output),
                "jax": finite_beta_flags(jax_output),
                "comparison": compare_fields(fortran_output, jax_output, FINITE_BETA_FLAGS),
            },
            "gradient_conversion": {
                "fortran": gradient_conversion_audit(fortran_output),
                "jax": gradient_conversion_audit(jax_output),
            },
            "current_fields": compare_fields(fortran_output, jax_output, CURRENT_FIELDS),
            "fsabjhat_assembly": {
                "fortran": fsabjhat_assembly_audit(fortran_output),
                "jax": fsabjhat_assembly_audit(jax_output),
            },
        }
        report["surfaces"][surface] = surface_report
    return report


def main() -> None:
    args = _build_parser().parse_args()
    report = run_audit(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        surface: {
            "worst_geometry_array_rel": max_ok_relative_error(payload["geometry_arrays"]),
            "worst_gradient_field_rel": max_ok_relative_error(payload["gradient_fields"]),
            "worst_current_scalar_rel": max_ok_relative_error(
                payload["current_fields"],
                ("FSABjHat", "FSABjHatOverB0", "FSABjHatOverRootFSAB2"),
            ),
            "fortran_current_assembly_status": payload["fsabjhat_assembly"]["fortran"].get("status"),
            "jax_current_assembly_status": payload["fsabjhat_assembly"]["jax"].get("status"),
        }
        for surface, payload in report["surfaces"].items()
    }
    print(json.dumps(summary if not args.json else _jsonable(report), indent=2, sort_keys=True))
    print(f"Wrote audit report to {args.out}")


if __name__ == "__main__":
    main()
