#!/usr/bin/env python
"""Paper-backed SFINCS_JAX / Redl comparison with an optional Fortran overlay.

This example uses the quasi-symmetric configurations and SFINCS benchmark decks
from the Zenodo artifact associated with arXiv:2205.02914.  It evaluates the
Redl bootstrap-current formula on the archived VMEC ``wout`` file and overlays
SFINCS_JAX RHSMode=1 solves at the same normalized toroidal fluxes. If the
archived SFINCS Fortran v3 ``sfincsOutput.h5`` files are available, the script
can also overlay them as a reference curve; no local Fortran executable is
required. Use ``--jax-vs-redl`` for a plot that does not load or require any
SFINCS Fortran v3 outputs.

The script deliberately makes one plot only: SFINCS_JAX points, the Redl
analytic curve, and optionally archived SFINCS Fortran v3 points. It does not
use NTX or NEOPAX data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file
from scipy.interpolate import interp1d

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.io import write_sfincs_jax_output_h5  # noqa: E402


ELEMENTARY_CHARGE = 1.602177e-19
SFINCS_PAPER_CURRENT_FACTOR = 437695.0 * 1.0e20 * ELEMENTARY_CHARGE
DEFAULT_ZENODO_ROOT = Path("/Users/rogeriojorge/local/20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current")
BENCHMARK_REL = Path("calculations/20211226-01-sfincs_for_precise_QS_for_Redl_benchmark")
DEFAULT_S_VALUES = "all"
QUICK_S_VALUES = "0.3,0.5,0.7"


@dataclass(frozen=True)
class BenchmarkCase:
    key: str
    title: str
    case_dir: str
    wout_name: str
    helicity_n: int


CASES = {
    "QA": BenchmarkCase(
        key="QA",
        title="Quasi-axisymmetric benchmark",
        case_dir="20211226-01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces",
        wout_name="wout_new_QA_aScaling.nc",
        helicity_n=0,
    ),
    "QH": BenchmarkCase(
        key="QH",
        title="Quasi-helical benchmark",
        case_dir="20211226-01-019_QH_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces",
        wout_name="wout_new_QH_aScaling.nc",
        helicity_n=-1,
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=tuple(CASES), default="QA", help="Zenodo benchmark configuration.")
    parser.add_argument(
        "--zenodo-root",
        type=Path,
        default=Path(os.environ.get("SFINCS_JAX_QS_ZENODO_ROOT", DEFAULT_ZENODO_ROOT)),
        help="Local Zenodo artifact root for arXiv:2205.02914.",
    )
    parser.add_argument(
        "--vmec-jax-root",
        type=Path,
        default=Path(os.environ.get("SFINCS_JAX_VMEC_JAX_ROOT", "/Users/rogeriojorge/local/vmec_jax")),
        help="Local vmec_jax checkout used for the Redl algebra.",
    )
    parser.add_argument(
        "--s-values",
        default=DEFAULT_S_VALUES,
        help="Comma-separated psi_N=s surfaces for SFINCS_JAX solves, or 'all' for every archived psiN_* deck.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Use a three-surface smoke set ({QUICK_S_VALUES}) instead of the default whole-radius scan.",
    )
    parser.add_argument("--redl-points", type=int, default=39, help="Number of Redl curve points.")
    parser.add_argument("--ntheta", type=int, default=13)
    parser.add_argument("--nzeta", type=int, default=13)
    parser.add_argument("--nxi", type=int, default=21)
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument(
        "--with-errorbars",
        action="store_true",
        help=(
            "Run/read two additional convergence probes and plot numerical error bars: "
            "one angular real-space refinement and one velocity-space refinement."
        ),
    )
    parser.add_argument("--real-ntheta", type=int, default=None, help="Real-space refinement Ntheta. Defaults to Ntheta+2.")
    parser.add_argument("--real-nzeta", type=int, default=None, help="Real-space refinement Nzeta. Defaults to Nzeta+2.")
    parser.add_argument("--velocity-nxi", type=int, default=None, help="Velocity-space refinement Nxi. Defaults to Nxi+4.")
    parser.add_argument("--velocity-nx", type=int, default=None, help="Velocity-space refinement Nx. Defaults to Nx+1.")
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-6)
    parser.add_argument(
        "--solve-method",
        default=None,
        help="Optional SFINCS_JAX RHSMode=1 solve method, e.g. sparse_pc_gmres for large production-sized runs.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun existing SFINCS_JAX outputs.")
    parser.add_argument(
        "--verbose-sfincs",
        action="store_true",
        help=(
            "Forward SFINCS_JAX phase/progress logging while running each kinetic solve. "
            "This is useful for production-grid probes that would otherwise look stalled."
        ),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Record failed SFINCS_JAX points and continue the radial scan instead of stopping at the first failure.",
    )
    parser.add_argument("--skip-sfincs", action="store_true", help="Only evaluate Redl and any existing SFINCS_JAX outputs.")
    parser.add_argument(
        "--hide-fortran",
        "--jax-vs-redl",
        "--sfincs-jax-vs-redl",
        dest="hide_fortran",
        action="store_true",
        help=(
            "Plot only SFINCS_JAX and Redl. This mode does not load or require "
            "archived SFINCS Fortran v3 sfincsOutput.h5 files."
        ),
    )
    parser.add_argument(
        "--fortran-case-root",
        type=Path,
        default=None,
        help=(
            "Optional case root containing SFINCS Fortran v3 psiN_*/sfincsOutput.h5 "
            "files to overlay instead of the archived Zenodo full-resolution root. "
            "Use this for reduced-resolution apples-to-apples reruns."
        ),
    )
    parser.add_argument(
        "--match-fortran-resolution",
        action="store_true",
        help=(
            "Set the SFINCS_JAX grid to the unique archived SFINCS Fortran v3 "
            "grid used on the selected surfaces before running solves."
        ),
    )
    parser.add_argument(
        "--require-same-resolution",
        action="store_true",
        help=(
            "Fail before plotting if SFINCS_JAX and the archived SFINCS Fortran "
            "v3 overlay are not using the same Ntheta,Nzeta,Nxi,Nx grid."
        ),
    )
    parser.add_argument(
        "--fortran-errorbar-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON sidecar with SFINCS Fortran v3 numerical error bars "
            "in SI units. Accepted forms are {'surfaces': [{'s': 0.5, "
            "'jdotb_si_errorbar': ...}]} or {'0.5': ...}."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/qs_paper_sfincs_jax_redl"))
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=Path("docs/_static/figures/vmec_jax_finite_beta"),
        help="Directory for PNG/PDF/JSON figure artifacts.",
    )
    parser.add_argument("--stem", default="qs_paper_sfincs_jax_redl_comparison")
    parser.add_argument(
        "--from-summary-json",
        type=Path,
        default=None,
        help=(
            "Regenerate PNG/PDF/JSON artifacts from an existing summary JSON. "
            "This avoids rerunning kinetic solves or requiring ignored HDF5 sidecar outputs."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Print the JSON summary.")
    return parser


def _require_vmec_jax(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if (root / "vmec_jax" / "__init__.py").exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from vmec_jax.redl_bootstrap import trapped_fraction_from_modb_sqrtg, redl_bootstrap_jdotb
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit("This example requires vmec_jax. Pass --vmec-jax-root or set SFINCS_JAX_VMEC_JAX_ROOT.") from exc
    return {
        "trapped_fraction_from_modb_sqrtg": trapped_fraction_from_modb_sqrtg,
        "redl_bootstrap_jdotb": redl_bootstrap_jdotb,
        "root": root,
    }


def _available_surface_values(case_root: Path) -> np.ndarray:
    values: list[float] = []
    for input_path in sorted(case_root.glob("psiN_*/input.namelist")):
        try:
            values.append(float(input_path.parent.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    if not values:
        raise ValueError(f"No archived psiN_* input decks found under {case_root}")
    return np.asarray(sorted(set(values)), dtype=float)


def _parse_s_values(text: str, *, case_root: Path | None = None) -> np.ndarray:
    token = str(text).strip().lower()
    if token in {"all", "archived", "whole", "full"}:
        if case_root is None:
            raise ValueError("--s-values all requires a case_root.")
        return _available_surface_values(case_root)
    values = np.asarray([float(piece.strip()) for piece in str(text).split(",") if piece.strip()], dtype=float)
    if values.size == 0:
        raise ValueError("--s-values must contain at least one value.")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("All --s-values must satisfy 0 < s < 1.")
    return np.asarray(sorted(set(float(v) for v in values)), dtype=float)


def _surface_dir_name(s_value: float) -> str:
    return f"psiN_{float(s_value):.12g}"


def _format_surface_label(s_value: float) -> str:
    return f"s{float(s_value):.3f}".replace(".", "p")


def _replace_namelist_scalar(text: str, key: str, value: int | float) -> str:
    pattern = re.compile(rf"(?im)^(\s*{re.escape(key)}\s*=\s*)([^\s!]+)(.*)$")
    replacement = rf"\g<1>{value}\g<3>"
    new_text, count = pattern.subn(replacement, text, count=1)
    if count == 0:
        raise ValueError(f"Could not find namelist scalar {key!r} to patch.")
    return new_text


def _write_patched_input(*, source: Path, destination: Path, ntheta: int, nzeta: int, nxi: int, nx: int, tolerance: float) -> None:
    text = source.read_text(encoding="utf-8")
    for key, value in (
        ("Ntheta", int(ntheta)),
        ("Nzeta", int(nzeta)),
        ("Nxi", int(nxi)),
        ("Nx", int(nx)),
        ("solverTolerance", float(tolerance)),
    ):
        text = _replace_namelist_scalar(text, key, value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _load_wout_arrays(wout_path: Path) -> dict[str, Any]:
    with netcdf_file(wout_path, "r", mmap=False) as ds:
        names = ["nfp", "ns", "phi", "iotas", "bvco", "buco", "bmnc", "gmnc", "xm_nyq", "xn_nyq"]
        data = {name: np.asarray(ds.variables[name].data).copy() for name in names}
    data["nfp"] = int(np.asarray(data["nfp"]).item())
    data["ns"] = int(np.asarray(data["ns"]).item())
    return data


def _redl_geometry_from_wout(*, vmec: dict[str, Any], wout_path: Path, surfaces: np.ndarray) -> dict[str, Any]:
    wout = _load_wout_arrays(wout_path)
    nfp = int(wout["nfp"])
    ns = int(wout["ns"])
    s_full = np.linspace(0.0, 1.0, ns)
    ds = s_full[1] - s_full[0]
    s_half = s_full[1:] - 0.5 * ds
    iota = interp1d(s_half, wout["iotas"][1:], fill_value="extrapolate")(surfaces)
    G = interp1d(s_half, wout["bvco"][1:], fill_value="extrapolate")(surfaces)
    current_I = interp1d(s_half, wout["buco"][1:], fill_value="extrapolate")(surfaces)
    bmnc = interp1d(s_half, wout["bmnc"][1:, :], axis=0, fill_value="extrapolate")(surfaces)
    gmnc = interp1d(s_half, wout["gmnc"][1:, :], axis=0, fill_value="extrapolate")(surfaces)

    ntheta = 64
    nphi = 65
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / nfp, nphi, endpoint=False)
    phi2d, theta2d = np.meshgrid(phi, theta, indexing="xy")
    modb = np.zeros((len(surfaces), ntheta, nphi), dtype=float)
    sqrtg = np.zeros_like(modb)
    xm = np.asarray(wout["xm_nyq"], dtype=float)
    xn = np.asarray(wout["xn_nyq"], dtype=float)
    for mode in range(len(xm)):
        cos_part = np.cos(xm[mode] * theta2d - xn[mode] * phi2d)
        modb += bmnc[:, mode, None, None] * cos_part[None, :, :]
        sqrtg += gmnc[:, mode, None, None] * cos_part[None, :, :]

    geom = vmec["trapped_fraction_from_modb_sqrtg"](modB=modb, sqrtg=sqrtg, n_lambda=32)
    fsa_1overb = np.asarray(geom["fsa_1overB"], dtype=float)
    R = (G + iota * current_I) * fsa_1overb
    return {
        "s": np.asarray(surfaces, dtype=float),
        "nfp": nfp,
        "psi_edge": -float(np.asarray(wout["phi"])[-1]) / (2.0 * np.pi),
        "G": np.asarray(G, dtype=float),
        "I": np.asarray(current_I, dtype=float),
        "R": np.asarray(R, dtype=float),
        "iota": np.asarray(iota, dtype=float),
        "epsilon": np.asarray(geom["epsilon"], dtype=float),
        "f_t": np.asarray(geom["f_t"], dtype=float),
    }


def _evaluate_redl(*, vmec: dict[str, Any], wout_path: Path, case: BenchmarkCase, surfaces: np.ndarray) -> np.ndarray:
    geom = _redl_geometry_from_wout(vmec=vmec, wout_path=wout_path, surfaces=surfaces)
    jdotb, _details = vmec["redl_bootstrap_jdotb"](
        s=geom["s"],
        G=geom["G"],
        R=geom["R"],
        iota=geom["iota"],
        epsilon=geom["epsilon"],
        f_t=geom["f_t"],
        psi_edge=geom["psi_edge"],
        nfp=geom["nfp"],
        helicity_n=case.helicity_n,
        ne_coeffs=4.13e20 * np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, -1.0]),
        Te_coeffs=12.0e3 * np.asarray([1.0, -1.0]),
        Zeff_coeffs=np.asarray([1.0]),
    )
    return np.asarray(jdotb, dtype=float)


def _read_scalar_h5(h5: h5py.File, key: str, default: float | int | None = None) -> float | int | None:
    if key not in h5:
        return default
    value = np.asarray(h5[key]).reshape(-1)[0]
    if isinstance(default, int):
        return int(value)
    return float(value)


def _read_jsonable_h5_scalar(h5: h5py.File, key: str) -> Any:
    """Read a scalar HDF5 diagnostic into a JSON-friendly Python value."""
    value = np.asarray(h5[key]).reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value.item() if hasattr(value, "item") else value


def _solver_diagnostics_from_h5(h5: h5py.File) -> dict[str, Any]:
    """Return solver diagnostics written by sfincs_jax output generation."""
    diagnostics: dict[str, Any] = {}
    for key in sorted(h5.keys()):
        if key.startswith("linearSolver"):
            diagnostics[key] = _read_jsonable_h5_scalar(h5, key)
    return diagnostics


def _configure_benchmark_solver_defaults() -> None:
    """Use the bounded non-autodiff solver lane for this paper comparison.

    The archived QS paper decks are RHSMode=1 full-FP bootstrap-current solves.
    For command-line benchmark/plot generation we want the fast host sparse
    route, not the differentiable implicit path. Environment defaults remain
    overridable for solver research runs.
    """
    os.environ.setdefault("SFINCS_JAX_IMPLICIT_SOLVE", "0")
    os.environ.setdefault("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    os.environ.setdefault("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    os.environ.setdefault(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_global_field_split_schur",
    )
    os.environ.setdefault("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_REQUIRED", "1")
    os.environ.setdefault("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB", "2048")
    os.environ.setdefault("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")


def _last_float_match(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_float_match(pattern: str, text: str) -> float | None:
    values: list[float] = []
    for match in re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
        value = match[-1] if isinstance(match, tuple) else match
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return max(values) if values else None


def _load_fortran_slurm_profile(surface_dir: Path) -> dict[str, Any]:
    """Parse archived/local Fortran v3/MUMPS timing and memory diagnostics.

    The Zenodo HDF5 files set ``elapsed time (s)`` to zero on these surfaces, but
    the adjacent Slurm logs include the SFINCS solve wall time and MUMPS memory
    accounting.  Slurm MaxRSS is not used here because the archived MPI job
    accounting reports unrealistically small per-task RSS for these runs.
    """

    log_paths = sorted(surface_dir.glob("slurm-*.out"))
    profile_path = log_paths[-1] if log_paths else surface_dir / "sfincs_fortran_stdout.txt"
    if not profile_path.exists():
        return {}
    text = profile_path.read_text(encoding="utf-8", errors="replace")
    stderr_path = surface_dir / "sfincs_fortran_stderr.txt"
    if stderr_path.exists():
        text += "\n" + stderr_path.read_text(encoding="utf-8", errors="replace")
    solve_s = _last_float_match(r"Done with the main solve\.\s*Time to solve:\s*([0-9.eE+-]+)", text)
    analysis_s = _last_float_match(r"Elapsed time in analysis driver=\s*([0-9.eE+-]+)", text)
    factor_s = _last_float_match(r"Elapsed time in factorization driver=\s*([0-9.eE+-]+)", text)
    solve_driver_s = _last_float_match(r"Elapsed time in solve driver=\s*([0-9.eE+-]+)", text)
    memory_allocated_total_mb = _max_float_match(
        r"(?:INFOG\(19\).*?:|Memory allocated,\s*total in Mbytes\s*\(INFOG\(19\)\):)\s*([0-9.eE+-]+)",
        text,
    )
    memory_effective_total_mb = _max_float_match(
        r"(?:INFOG\(22\).*?:|Memory effectively used,\s*total in Mbytes\s*\(INFOG\(22\)\):)\s*([0-9.eE+-]+)",
        text,
    )
    memory_allocated_max_mb = _max_float_match(
        r"(?:INFOG\(18\).*?:|Memory allocated,\s*max in Mbytes\s*\(INFOG\(18\)\):)\s*([0-9.eE+-]+)",
        text,
    )
    memory_effective_max_mb = _max_float_match(
        r"(?:INFOG\(21\).*?:|Memory effectively used,\s*max in\s*Mbytes\s*\(INFOG\(21\)\):)\s*([0-9.eE+-]+)",
        text,
    )
    mumps_solve_space_mb = _max_float_match(r"Space in MBYTES used for solve\s*:\s*([0-9.eE+-]+)", text)
    max_rss_bytes = _max_float_match(r"^\s*([0-9.eE+-]+)\s+maximum resident set size", text)
    max_rss_mb = None if max_rss_bytes is None else float(max_rss_bytes) / 1.0e6
    memory_mb = memory_effective_total_mb
    memory_metric = "mumps_effective_total_mb"
    if memory_mb is None:
        memory_mb = memory_allocated_total_mb
        memory_metric = "mumps_allocated_total_mb"
    if memory_mb is None:
        memory_mb = mumps_solve_space_mb
        memory_metric = "mumps_solve_space_mb"
    if memory_mb is None:
        memory_mb = max_rss_mb
        memory_metric = "process_max_rss_mb"
    return {
        "profile_log": str(profile_path),
        "profile_stderr": str(stderr_path) if stderr_path.exists() else None,
        "elapsed_s": solve_s,
        "analysis_s": analysis_s,
        "factor_s": factor_s,
        "solve_driver_s": solve_driver_s,
        "memory_mb": memory_mb,
        "memory_metric": memory_metric,
        "mumps_memory_allocated_total_mb": memory_allocated_total_mb,
        "mumps_memory_effective_total_mb": memory_effective_total_mb,
        "mumps_memory_allocated_max_rank_mb": memory_allocated_max_mb,
        "mumps_memory_effective_max_rank_mb": memory_effective_max_mb,
        "mumps_solve_space_mb": mumps_solve_space_mb,
        "max_rss_mb": max_rss_mb,
    }


def _load_archived_fortran_outputs(*, case_root: Path) -> list[dict[str, Any]]:
    r"""Load archived SFINCS Fortran v3 bootstrap-current outputs.

    The Zenodo benchmark stores one ``sfincsOutput.h5`` per ``psiN_*`` surface.
    Those outputs use the same ``FSABjHat`` normalization as the original paper
    plotting scripts, so the same archived scale factor converts them to
    :math:`\langle J\cdot B\rangle` in SI units.
    """
    rows: list[dict[str, Any]] = []
    for output_path in sorted(case_root.glob("psiN_*/sfincsOutput.h5")):
        try:
            s_value = float(output_path.parent.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        with h5py.File(output_path, "r") as h5:
            fsab_j_hat = float(np.asarray(h5["FSABjHat"]).reshape(-1)[0])
            elapsed_h5 = _read_scalar_h5(h5, "elapsed time (s)", default=None)
            slurm_profile = _load_fortran_slurm_profile(output_path.parent)
            row = {
                "status": "ok",
                "s": s_value,
                "output_path": str(output_path),
                "FSABjHat": fsab_j_hat,
                "jdotb_si": fsab_j_hat * SFINCS_PAPER_CURRENT_FACTOR,
                "Ntheta": _read_scalar_h5(h5, "Ntheta"),
                "Nzeta": _read_scalar_h5(h5, "Nzeta"),
                "Nxi": _read_scalar_h5(h5, "Nxi"),
                "Nx": _read_scalar_h5(h5, "Nx"),
                "NIterations": _read_scalar_h5(h5, "NIterations", default=0),
                "elapsed_s": (
                    float(slurm_profile["elapsed_s"])
                    if slurm_profile.get("elapsed_s") is not None
                    else (float(elapsed_h5) if elapsed_h5 not in {None, 0.0} else None)
                ),
                "memory_mb": slurm_profile.get("memory_mb"),
                "memory_metric": slurm_profile.get("memory_metric"),
                "profile": slurm_profile,
            }
        rows.append(row)
    return sorted(rows, key=lambda row: float(row["s"]))


def _solver_memory_mb_from_diagnostics(diagnostics: dict[str, Any]) -> float | None:
    factor_nbytes = diagnostics.get("linearSolverSparsePCFactorNbytesEstimate")
    csr_nbytes = diagnostics.get("linearSolverCsrOperatorNbytes")
    values = []
    for raw in (factor_nbytes, csr_nbytes):
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return float(sum(values) / 1.0e6)


def _verbose_sfincs_enabled(args: argparse.Namespace) -> bool:
    env_value = os.environ.get("SFINCS_JAX_EXAMPLE_VERBOSE", "").strip().lower()
    env_enabled = env_value in {"1", "true", "t", "yes", "on", ".true.", ".t."}
    return bool(getattr(args, "verbose_sfincs", False) or env_enabled)


def _run_or_read_sfincs_jax(
    *,
    source_input: Path,
    wout_path: Path,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path = run_dir / "sfincsOutput.h5"
    input_path = run_dir / "input.namelist"
    metadata_path = run_dir / "sfincs_jax_run_metadata.json"
    cached = output_path.exists() and not args.force
    if not cached:
        if args.skip_sfincs:
            return {"status": "missing", "output_path": str(output_path)}
        _configure_benchmark_solver_defaults()
        _write_patched_input(
            source=source_input,
            destination=input_path,
            ntheta=args.ntheta,
            nzeta=args.nzeta,
            nxi=args.nxi,
            nx=args.nx,
            tolerance=args.solver_tolerance,
        )
        t0 = time.perf_counter()
        write_sfincs_jax_output_h5(
            input_namelist=input_path,
            output_path=output_path,
            wout_path=wout_path,
            overwrite=True,
            compute_solution=True,
            solve_method=args.solve_method,
            differentiable=False,
            verbose=_verbose_sfincs_enabled(args),
        )
        elapsed = time.perf_counter() - t0
    else:
        elapsed = None

    metadata: dict[str, Any] = {}
    if cached and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}

    with h5py.File(output_path, "r") as h5:
        fsab_j_hat = float(np.asarray(h5["FSABjHat"]).reshape(-1)[0])
        r_n = float(np.asarray(h5["rN"]).reshape(-1)[0]) if "rN" in h5 else float("nan")
        psi_n = float(np.asarray(h5["psiN"]).reshape(-1)[0]) if "psiN" in h5 else float("nan")
        n_iterations = int(np.asarray(h5["NIterations"]).reshape(-1)[0]) if "NIterations" in h5 else None
        solver_diagnostics = _solver_diagnostics_from_h5(h5)
        memory_mb = _solver_memory_mb_from_diagnostics(solver_diagnostics)
    row = {
        "status": "ok",
        "output_path": str(output_path),
        "elapsed_s": elapsed if elapsed is not None else metadata.get("elapsed_s"),
        "cached": bool(cached),
        "FSABjHat": fsab_j_hat,
        "jdotb_si": fsab_j_hat * SFINCS_PAPER_CURRENT_FACTOR,
        "rN": r_n,
        "psiN": psi_n,
        "NIterations": n_iterations,
        "solver_diagnostics": solver_diagnostics,
        "memory_mb": memory_mb,
        "memory_metric": "linear_solver_factor_plus_csr_estimate_mb",
    }
    if not cached:
        stored_metadata = {
            "elapsed_s": elapsed,
            "Ntheta": int(args.ntheta),
            "Nzeta": int(args.nzeta),
            "Nxi": int(args.nxi),
            "Nx": int(args.nx),
            "solverTolerance": float(args.solver_tolerance),
            "solveMethod": None if args.solve_method is None else str(args.solve_method),
            "differentiable": False,
            "output_path": str(output_path),
            "solver_diagnostics": solver_diagnostics,
        }
        metadata_path.write_text(
            json.dumps(stored_metadata, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    return row


def _scan_args_with_resolution(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    ntheta: int,
    nzeta: int,
    nxi: int,
    nx: int,
) -> argparse.Namespace:
    run_args = copy(args)
    run_args.out_dir = out_dir
    run_args.ntheta = int(ntheta)
    run_args.nzeta = int(nzeta)
    run_args.nxi = int(nxi)
    run_args.nx = int(nx)
    return run_args


def _resolution_dict(args: argparse.Namespace) -> dict[str, int | float | str | None]:
    return {
        "Ntheta": int(args.ntheta),
        "Nzeta": int(args.nzeta),
        "Nxi": int(args.nxi),
        "Nx": int(args.nx),
        "solverTolerance": float(args.solver_tolerance),
        "solveMethod": None if args.solve_method is None else str(args.solve_method),
    }


def _resolution_tuple_from_args(args: argparse.Namespace) -> tuple[int, int, int, int]:
    return (int(args.ntheta), int(args.nzeta), int(args.nxi), int(args.nx))


def _resolution_tuple_from_row(row: dict[str, Any]) -> tuple[int, int, int, int] | None:
    keys = ("Ntheta", "Nzeta", "Nxi", "Nx")
    if any(row.get(key) is None for key in keys):
        return None
    return tuple(int(row[key]) for key in keys)  # type: ignore[return-value]


def _resolution_label(resolution: tuple[int, int, int, int] | None) -> str | None:
    if resolution is None:
        return None
    return "x".join(str(int(value)) for value in resolution)


def _selected_fortran_rows(
    *,
    s_values: np.ndarray,
    fortran_by_surface: dict[float, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for s_value in s_values:
        row = fortran_by_surface.get(round(float(s_value), 12))
        if row is not None and row.get("status") == "ok":
            rows.append(row)
    return rows


def _unique_resolution_tuples(rows: list[dict[str, Any]]) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    ordered: list[tuple[int, int, int, int]] = []
    for row in rows:
        resolution = _resolution_tuple_from_row(row)
        if resolution is None or resolution in seen:
            continue
        ordered.append(resolution)
        seen.add(resolution)
    return ordered


def _apply_matching_fortran_resolution(args: argparse.Namespace, selected_fortran_rows: list[dict[str, Any]]) -> None:
    """Mutate ``args`` so the JAX grid matches the selected Fortran reference.

    A public SFINCS_JAX-vs-Fortran plot should not silently compare reduced JAX
    solves against full-resolution archived Fortran outputs.  The Zenodo QA/QH
    benchmark has one grid on all surfaces, but the helper is intentionally
    strict so mixed Fortran grids require an explicit user decision.
    """

    if not selected_fortran_rows:
        raise SystemExit("--match-fortran-resolution requires archived Fortran v3 rows on the selected surfaces.")
    resolutions = _unique_resolution_tuples(selected_fortran_rows)
    if len(resolutions) != 1:
        labels = ", ".join(str(item) for item in resolutions) or "unknown"
        raise SystemExit(f"--match-fortran-resolution requires one unique Fortran grid; found {labels}.")
    args.ntheta, args.nzeta, args.nxi, args.nx = resolutions[0]


def _resolution_comparison(
    *,
    args: argparse.Namespace,
    selected_fortran_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    jax_resolution = _resolution_tuple_from_args(args)
    fortran_resolutions = _unique_resolution_tuples(selected_fortran_rows)
    same_resolution = bool(fortran_resolutions) and len(fortran_resolutions) == 1 and fortran_resolutions[0] == jax_resolution
    if not selected_fortran_rows:
        status = "no_fortran_overlay"
    elif same_resolution:
        status = "same_resolution"
    else:
        status = "mixed_resolution"
    return {
        "status": status,
        "same_resolution_on_compared_surfaces": bool(same_resolution),
        "sfincs_jax_resolution": {
            "Ntheta": jax_resolution[0],
            "Nzeta": jax_resolution[1],
            "Nxi": jax_resolution[2],
            "Nx": jax_resolution[3],
        },
        "sfincs_jax_resolution_label": _resolution_label(jax_resolution),
        "sfincs_fortran_v3_resolutions": [
            {"Ntheta": item[0], "Nzeta": item[1], "Nxi": item[2], "Nx": item[3]} for item in fortran_resolutions
        ],
        "sfincs_fortran_v3_resolution_label": (
            _resolution_label(fortran_resolutions[0]) if len(fortran_resolutions) == 1 else None
        ),
        "compared_fortran_points": int(len(selected_fortran_rows)),
        "claim": (
            "SFINCS_JAX and SFINCS Fortran v3 are compared on the same radial surfaces and resolution."
            if same_resolution
            else (
                "No SFINCS Fortran v3 overlay was loaded; this payload is a SFINCS_JAX-vs-Redl diagnostic."
                if not selected_fortran_rows
                else "Do not use JAX-vs-Fortran differences from this payload as a same-resolution parity claim."
            )
        ),
    }


def _enforce_same_resolution_requirement(args: argparse.Namespace, comparison: dict[str, Any]) -> None:
    if not bool(args.require_same_resolution):
        return
    if bool(args.hide_fortran):
        return
    if comparison.get("status") == "no_fortran_overlay":
        raise SystemExit("--require-same-resolution needs archived Fortran v3 rows on the selected surfaces.")
    if not bool(comparison.get("same_resolution_on_compared_surfaces")):
        jax_label = comparison.get("sfincs_jax_resolution_label")
        fortran_label = comparison.get("sfincs_fortran_v3_resolution_label") or "mixed/unknown"
        raise SystemExit(
            "Refusing mixed-resolution SFINCS_JAX/SFINCS Fortran v3 figure: "
            f"JAX grid is {jax_label}, Fortran grid is {fortran_label}. "
            "Use --match-fortran-resolution or remove --require-same-resolution."
        )


def _load_fortran_errorbar_map(path: Path | None) -> dict[float, float]:
    """Load optional archived-Fortran uncertainty bars without inventing them."""

    if path is None:
        return {}
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    rows: list[dict[str, Any]]
    if isinstance(data, list):
        rows = [row for row in data if isinstance(row, dict)]
    elif isinstance(data, dict) and isinstance(data.get("surfaces"), list):
        rows = [row for row in data["surfaces"] if isinstance(row, dict)]
    elif isinstance(data, dict):
        result: dict[float, float] = {}
        for key, value in data.items():
            if key in {"definition", "source", "units"}:
                continue
            try:
                result[round(float(key), 12)] = float(value)
            except (TypeError, ValueError):
                continue
        return result
    else:
        raise ValueError(f"Unsupported Fortran errorbar JSON structure: {path}")

    result = {}
    for row in rows:
        if "s" not in row:
            continue
        raw_error = row.get("jdotb_si_errorbar", row.get("errorbar", row.get("errorbar_si")))
        if raw_error is None:
            continue
        result[round(float(row["s"]), 12)] = float(raw_error)
    return result


def _apply_fortran_errorbars(rows: list[dict[str, Any]], errorbars: dict[float, float]) -> None:
    for row in rows:
        value = errorbars.get(round(float(row["s"]), 12))
        if value is not None:
            row["jdotb_si_errorbar"] = float(value)


def _run_surface_scan(
    *,
    args: argparse.Namespace,
    case: BenchmarkCase,
    case_root: Path,
    wout_path: Path,
    s_values: np.ndarray,
    redl_at_sfincs: np.ndarray,
    fortran_by_surface: dict[float, dict[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    out_dir = args.out_dir / case.key
    sfincs_rows: list[dict[str, Any]] = []
    for s_value, redl_value in zip(s_values, redl_at_sfincs, strict=True):
        source_input = case_root / _surface_dir_name(float(s_value)) / "input.namelist"
        if not source_input.exists():
            raise FileNotFoundError(f"Missing archived input for s={s_value}: {source_input}")
        run_dir = out_dir / _format_surface_label(float(s_value)) / f"Ntheta{args.ntheta}_Nzeta{args.nzeta}_Nxi{args.nxi}_Nx{args.nx}"
        print(
            f"[{len(sfincs_rows) + 1:02d}/{len(s_values):02d}] {label} "
            f"s={float(s_value):.6g} grid={args.ntheta}x{args.nzeta}x{args.nxi}x{args.nx}",
            flush=True,
        )
        try:
            row = _run_or_read_sfincs_jax(
                source_input=source_input,
                wout_path=wout_path,
                run_dir=run_dir,
                args=args,
            )
        except Exception as exc:
            if not args.keep_going:
                raise
            row = {
                "status": "error",
                "output_path": str(run_dir / "sfincsOutput.h5"),
                "error": str(exc),
                "cached": False,
            }
        row.update({"s": float(s_value), "redl_jdotb_si": float(redl_value)})
        if row["status"] == "ok":
            row["relative_difference_vs_redl"] = _relative_difference(float(row["jdotb_si"]), float(redl_value))
            fortran_row = fortran_by_surface.get(round(float(s_value), 12))
            if fortran_row is not None:
                row["fortran_jdotb_si"] = float(fortran_row["jdotb_si"])
                row["relative_difference_vs_fortran"] = _relative_difference(float(row["jdotb_si"]), float(fortran_row["jdotb_si"]))
            elapsed_label = "cached" if row.get("cached") else f"{float(row.get('elapsed_s', 0.0)):.2f} s"
            print(
                f"    ok: {elapsed_label}, rel(JAX,Redl)={row['relative_difference_vs_redl']:.3e}",
                flush=True,
            )
        else:
            print(f"    {row['status']}: {row.get('error', row.get('output_path'))}", flush=True)
        sfincs_rows.append(row)
    return sfincs_rows


def _sfincs_jdotb_profile_from_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [np.nan if row.get("status") != "ok" else float(row["jdotb_si"]) for row in rows],
        dtype=float,
    )


def _pointwise_max_abs_delta(reference: np.ndarray, candidates: list[np.ndarray]) -> np.ndarray:
    error = np.full_like(reference, np.nan, dtype=float)
    for idx, base in enumerate(reference):
        if not np.isfinite(base):
            continue
        deltas = [abs(float(candidate[idx]) - float(base)) for candidate in candidates if idx < len(candidate) and np.isfinite(candidate[idx])]
        if deltas:
            error[idx] = max(deltas)
    return error


def _run_convergence_errorbars(
    *,
    args: argparse.Namespace,
    case: BenchmarkCase,
    case_root: Path,
    wout_path: Path,
    s_values: np.ndarray,
    redl_at_sfincs: np.ndarray,
    fortran_by_surface: dict[float, dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not bool(args.with_errorbars):
        return None

    real_ntheta = int(args.real_ntheta) if args.real_ntheta is not None else int(args.ntheta) + 2
    real_nzeta = int(args.real_nzeta) if args.real_nzeta is not None else int(args.nzeta) + 2
    velocity_nxi = int(args.velocity_nxi) if args.velocity_nxi is not None else int(args.nxi) + 4
    velocity_nx = int(args.velocity_nx) if args.velocity_nx is not None else int(args.nx) + 1
    real_args = _scan_args_with_resolution(
        args,
        out_dir=args.out_dir / "real_space_refined",
        ntheta=real_ntheta,
        nzeta=real_nzeta,
        nxi=int(args.nxi),
        nx=int(args.nx),
    )
    velocity_args = _scan_args_with_resolution(
        args,
        out_dir=args.out_dir / "velocity_space_refined",
        ntheta=int(args.ntheta),
        nzeta=int(args.nzeta),
        nxi=velocity_nxi,
        nx=velocity_nx,
    )

    print(
        "Running/reading convergence probes: "
        f"real-space={_resolution_dict(real_args)}, velocity-space={_resolution_dict(velocity_args)}",
        flush=True,
    )
    real_rows = _run_surface_scan(
        args=real_args,
        case=case,
        case_root=case_root,
        wout_path=wout_path,
        s_values=s_values,
        redl_at_sfincs=redl_at_sfincs,
        fortran_by_surface=fortran_by_surface,
        label="real-space",
    )
    velocity_rows = _run_surface_scan(
        args=velocity_args,
        case=case,
        case_root=case_root,
        wout_path=wout_path,
        s_values=s_values,
        redl_at_sfincs=redl_at_sfincs,
        fortran_by_surface=fortran_by_surface,
        label="velocity-space",
    )
    baseline = _sfincs_jdotb_profile_from_rows(baseline_rows)
    real = _sfincs_jdotb_profile_from_rows(real_rows)
    velocity = _sfincs_jdotb_profile_from_rows(velocity_rows)
    real_delta = np.abs(real - baseline)
    velocity_delta = np.abs(velocity - baseline)
    error = _pointwise_max_abs_delta(baseline, [real, velocity])
    rel_to_baseline = error / np.maximum(np.abs(baseline), 1.0e-300)
    return {
        "enabled": True,
        "definition": (
            "Error bars are the pointwise maximum absolute change in the SFINCS-JAX "
            "SI bootstrap-current diagnostic when independently refining angular "
            "real-space resolution and velocity-space resolution from the baseline grid."
        ),
        "baseline_resolution": _resolution_dict(args),
        "real_space_resolution": _resolution_dict(real_args),
        "velocity_space_resolution": _resolution_dict(velocity_args),
        "jdotb_si_errorbar": [float(v) for v in error],
        "jdotb_si_errorbar_rel_to_baseline": [float(v) for v in rel_to_baseline],
        "real_space_delta_jdotb_si": [float(v) for v in real_delta],
        "velocity_space_delta_jdotb_si": [float(v) for v in velocity_delta],
        "real_space": real_rows,
        "velocity_space": velocity_rows,
    }


def _plot(payload: dict[str, Any], *, png_path: Path, pdf_path: Path) -> None:
    redl = payload["redl"]
    hide_fortran = bool(payload.get("hide_fortran", False))
    fortran = [] if hide_fortran else [row for row in payload.get("sfincs_fortran_v3", []) if row["status"] == "ok"]
    sfincs = [row for row in payload["sfincs_jax"] if row["status"] == "ok"]
    s_redl = np.asarray(redl["s"], dtype=float)
    j_redl = np.asarray(redl["jdotb_si"], dtype=float) / 1.0e6
    fig = plt.figure(figsize=(12.8, 5.35))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, width_ratios=(2.25, 1.0), wspace=0.32, hspace=0.42)
    ax = fig.add_subplot(gs[:, 0])
    ax_time = fig.add_subplot(gs[0, 1])
    ax_memory = fig.add_subplot(gs[1, 1])

    ax.plot(s_redl, j_redl, color="#1f5f99", lw=2.7, label="Redl analytic formula")
    if fortran:
        s_fortran = np.asarray([row["s"] for row in fortran], dtype=float)
        j_fortran = np.asarray([row["jdotb_si"] for row in fortran], dtype=float) / 1.0e6
        fortran_errorbar = np.asarray(
            [float(row["jdotb_si_errorbar"]) / 1.0e6 if row.get("jdotb_si_errorbar") is not None else np.nan for row in fortran],
            dtype=float,
        )
        ax.plot(
            s_fortran,
            j_fortran,
            "-",
            lw=2.2,
            color="#2d2d2d",
            marker="s",
            ms=4.0,
            markevery=max(len(s_fortran) // 12, 1),
            label="SFINCS Fortran v3",
        )
        if np.any(np.isfinite(fortran_errorbar)):
            mask = np.isfinite(fortran_errorbar)
            ax.errorbar(
                s_fortran[mask],
                j_fortran[mask],
                yerr=fortran_errorbar[mask],
                fmt="none",
                ecolor="#2d2d2d",
                elinewidth=1.2,
                capsize=3.0,
                alpha=0.82,
            )
    if sfincs:
        s_sfincs = np.asarray([row["s"] for row in sfincs], dtype=float)
        j_sfincs = np.asarray([row["jdotb_si"] for row in sfincs], dtype=float) / 1.0e6
        errorbar = np.full_like(j_sfincs, np.nan, dtype=float)
        convergence = payload.get("convergence_errorbars")
        if convergence is not None:
            raw = np.asarray(convergence.get("jdotb_si_errorbar", []), dtype=float) / 1.0e6
            if raw.shape == errorbar.shape:
                errorbar = raw
        if np.any(np.isfinite(errorbar)):
            ax.errorbar(
                s_sfincs,
                j_sfincs,
                yerr=errorbar,
                fmt="o",
                ms=8.5,
                capsize=3.5,
                color="#c43b2f",
                markeredgecolor="white",
                markeredgewidth=1.1,
                label="SFINCS_JAX (refinement bars)",
            )
        else:
            ax.plot(
                s_sfincs,
                j_sfincs,
                "o",
                ms=7.0,
                color="#c43b2f",
                markeredgecolor="white",
                markeredgewidth=0.9,
                label="SFINCS_JAX kinetic solve",
            )
    ax.set_xlabel("Normalized toroidal flux $s = \\psi_N$")
    ax.set_ylabel(r"$\langle \mathbf{J}\cdot\mathbf{B}\rangle$ [MA T m$^{-2}$]")
    ax.set_title(payload.get("plot_title", payload["case_title"]))
    ax.grid(True, color="#dddddd", lw=0.8, alpha=0.9)
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    metrics = payload.get("metrics", {})
    if metrics.get("max_jax_relative_difference_vs_redl") is not None:
        resolution_comparison = payload.get("resolution_comparison", {})
        same_resolution = resolution_comparison.get("same_resolution_on_compared_surfaces")
        fortran_resolution_label = resolution_comparison.get("sfincs_fortran_v3_resolution_label")
        lines = [
            f"JAX points = {metrics.get('completed_points', 0)}/{metrics.get('requested_points', metrics.get('completed_points', 0))}",
            f"JAX vs Redl max = {metrics['max_jax_relative_difference_vs_redl']:.2%}",
        ]
        if metrics.get("max_jax_relative_difference_vs_fortran") is not None:
            prefix = "JAX vs Fortran max"
            if same_resolution is False:
                prefix += " (mixed grids)"
            lines.append(f"{prefix} = {metrics['max_jax_relative_difference_vs_fortran']:.2%}")
        if metrics.get("max_fortran_relative_difference_vs_redl_on_jax_surfaces") is not None:
            lines.append(f"Fortran vs Redl max = {metrics['max_fortran_relative_difference_vs_redl_on_jax_surfaces']:.2%}")
        if metrics.get("sfincs_jax_elapsed_s_sum") is not None:
            lines.append(f"JAX solve wall = {metrics['sfincs_jax_elapsed_s_sum']:.1f} s")
        if metrics.get("max_errorbar_rel_to_baseline") is not None:
            lines.append(f"max refinement bar = {metrics['max_errorbar_rel_to_baseline']:.2%}")
        lines.append(f"JAX grid = {payload['sfincs_resolution_label']}")
        if fortran_resolution_label:
            lines.append(f"Fortran grid = {fortran_resolution_label}")
        if hide_fortran:
            lines.append("SFINCS_JAX vs Redl only")
        elif same_resolution is True:
            lines.append("same-resolution comparison")
        elif same_resolution is False:
            lines.append("mixed-grid diagnostic")
        ax.text(
            0.04,
            0.06,
            "\n".join(lines),
            transform=ax.transAxes,
            fontsize=8.8,
            ha="left",
            va="bottom",
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cccccc", "alpha": 0.92},
        )

    performance = payload.get("performance", {})
    runtime = performance.get("runtime_total_s", {})
    memory = performance.get("memory_peak_mb", {})
    if hide_fortran:
        labels = ["SFINCS_JAX"]
        colors = ["#c43b2f"]
        performance_keys = ["sfincs_jax"]
    else:
        labels = ["SFINCS_JAX", "Fortran v3"]
        colors = ["#c43b2f", "#2d2d2d"]
        performance_keys = ["sfincs_jax", "sfincs_fortran_v3"]

    def _panel_bars(panel_ax: Any, values: dict[str, Any], *, title: str, ylabel: str) -> None:
        raw_values = [values.get(key) for key in performance_keys]
        finite = [float(v) for v in raw_values if v is not None and np.isfinite(float(v)) and float(v) > 0.0]
        if not finite:
            panel_ax.text(0.5, 0.5, "metadata unavailable", ha="center", va="center", transform=panel_ax.transAxes)
            panel_ax.set_axis_off()
            return
        x_positions = np.arange(len(labels), dtype=float)
        bar_values = [float(v) if v is not None and np.isfinite(float(v)) and float(v) > 0.0 else np.nan for v in raw_values]
        panel_ax.bar(x_positions, bar_values, color=colors, width=0.62)
        use_log = max(finite) / max(min(finite), 1.0e-300) > 8.0
        if use_log:
            panel_ax.set_yscale("log")
            panel_ax.set_ylim(max(min(finite) * 0.65, 1.0e-12), max(finite) * 2.4)
        else:
            panel_ax.set_ylim(0.0, max(finite) * 1.28)
        for xpos, value in zip(x_positions, bar_values, strict=True):
            if not np.isfinite(value):
                panel_ax.text(xpos, max(finite), "n/a", ha="center", va="bottom", fontsize=8.5)
                continue
            panel_ax.text(xpos, value * 1.06, f"{value:.2g}", ha="center", va="bottom", fontsize=8.5)
        panel_ax.set_xticks(x_positions, labels, rotation=0)
        panel_ax.set_title(title, fontsize=10.5)
        panel_ax.set_ylabel(ylabel)
        panel_ax.grid(True, axis="y", color="#dddddd", lw=0.8, alpha=0.9)
        panel_ax.spines["top"].set_visible(False)
        panel_ax.spines["right"].set_visible(False)

    _panel_bars(ax_time, runtime, title="All plotted radii runtime", ylabel="wall time [s]")
    _panel_bars(ax_memory, memory, title="Peak solver memory", ylabel="memory [MB]")
    if performance.get("memory_definition"):
        ax_memory.text(
            0.0,
            -0.42,
            str(performance["memory_definition"]),
            transform=ax_memory.transAxes,
            fontsize=7.4,
            ha="left",
            va="top",
            color="#555555",
        )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _write_plot_artifacts(payload: dict[str, Any], *, fig_dir: Path, stem: str, print_json: bool = False) -> None:
    """Write the bootstrap-current figure bundle from an already-built payload."""

    fig_dir.mkdir(parents=True, exist_ok=True)
    png_path = fig_dir / f"{stem}.png"
    pdf_path = fig_dir / f"{stem}.pdf"
    json_path = fig_dir / f"{stem}.json"
    _plot(payload, png_path=png_path, pdf_path=pdf_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if print_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {json_path}")


def _write_plot_artifacts_from_summary_json(*, summary_json: Path, fig_dir: Path, stem: str, print_json: bool = False) -> None:
    payload = json.loads(summary_json.expanduser().read_text(encoding="utf-8"))
    _write_plot_artifacts(payload, fig_dir=fig_dir, stem=stem, print_json=print_json)


def _relative_difference(value: float, reference: float) -> float:
    return abs(float(value) - float(reference)) / max(abs(float(reference)), 1.0e-300)


def _lookup_by_surface(rows: list[dict[str, Any]]) -> dict[float, dict[str, Any]]:
    return {round(float(row["s"]), 12): row for row in rows}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.from_summary_json is not None:
        _write_plot_artifacts_from_summary_json(
            summary_json=args.from_summary_json,
            fig_dir=args.fig_dir,
            stem=args.stem,
            print_json=args.json,
        )
        return 0

    vmec = _require_vmec_jax(args.vmec_jax_root)
    case = CASES[args.case]
    zenodo_root = args.zenodo_root.expanduser().resolve()
    benchmark_root = zenodo_root / BENCHMARK_REL
    case_root = benchmark_root / case.case_dir
    wout_path = benchmark_root / case.wout_name
    if not case_root.exists():
        raise FileNotFoundError(f"Missing Zenodo case directory: {case_root}")
    if not wout_path.exists():
        raise FileNotFoundError(f"Missing Zenodo wout file: {wout_path}")

    requested_s_values = QUICK_S_VALUES if args.quick else args.s_values
    s_values = _parse_s_values(requested_s_values, case_root=case_root)
    redl_s = np.linspace(0.025, 0.975, int(args.redl_points))
    redl_curve = _evaluate_redl(vmec=vmec, wout_path=wout_path, case=case, surfaces=redl_s)
    redl_at_sfincs = _evaluate_redl(vmec=vmec, wout_path=wout_path, case=case, surfaces=s_values)
    fortran_case_root = case_root if args.fortran_case_root is None else args.fortran_case_root.expanduser().resolve()
    fortran_rows = [] if args.hide_fortran else _load_archived_fortran_outputs(case_root=fortran_case_root)
    fortran_by_surface = _lookup_by_surface(fortran_rows)
    selected_fortran_rows_requested = _selected_fortran_rows(s_values=s_values, fortran_by_surface=fortran_by_surface)
    if args.match_fortran_resolution:
        _apply_matching_fortran_resolution(args, selected_fortran_rows_requested)
    fortran_errorbars = _load_fortran_errorbar_map(args.fortran_errorbar_json)
    _apply_fortran_errorbars(fortran_rows, fortran_errorbars)
    resolution_comparison = _resolution_comparison(
        args=args,
        selected_fortran_rows=selected_fortran_rows_requested,
    )
    _enforce_same_resolution_requirement(args, resolution_comparison)
    if fortran_rows:
        fortran_s = np.asarray([row["s"] for row in fortran_rows], dtype=float)
        fortran_redl = _evaluate_redl(vmec=vmec, wout_path=wout_path, case=case, surfaces=fortran_s)
        for row, redl_value in zip(fortran_rows, fortran_redl, strict=True):
            row["redl_jdotb_si"] = float(redl_value)
            row["relative_difference_vs_redl"] = _relative_difference(float(row["jdotb_si"]), float(redl_value))

    sfincs_rows = _run_surface_scan(
        args=args,
        case=case,
        case_root=case_root,
        wout_path=wout_path,
        s_values=s_values,
        redl_at_sfincs=redl_at_sfincs,
        fortran_by_surface=fortran_by_surface,
        label="baseline",
    )
    convergence = _run_convergence_errorbars(
        args=args,
        case=case,
        case_root=case_root,
        wout_path=wout_path,
        s_values=s_values,
        redl_at_sfincs=redl_at_sfincs,
        fortran_by_surface=fortran_by_surface,
        baseline_rows=sfincs_rows,
    )

    ok_rows = [row for row in sfincs_rows if row["status"] == "ok"]
    ok_fortran_rows = [row for row in fortran_rows if row["status"] == "ok"]
    selected_fortran_rows = [fortran_by_surface[round(float(row["s"]), 12)] for row in ok_rows if round(float(row["s"]), 12) in fortran_by_surface]
    max_jax_redl = max((float(row["relative_difference_vs_redl"]) for row in ok_rows), default=None)
    max_jax_fortran = max((float(row["relative_difference_vs_fortran"]) for row in ok_rows if "relative_difference_vs_fortran" in row), default=None)
    same_resolution_fortran = bool(resolution_comparison.get("same_resolution_on_compared_surfaces"))
    max_fortran_redl = max((float(row["relative_difference_vs_redl"]) for row in ok_fortran_rows), default=None)
    max_selected_fortran_redl = max((float(row["relative_difference_vs_redl"]) for row in selected_fortran_rows), default=None)
    elapsed_values = [float(row["elapsed_s"]) for row in ok_rows if row.get("elapsed_s") is not None]
    jax_memory_values = [float(row["memory_mb"]) for row in ok_rows if row.get("memory_mb") is not None]
    selected_fortran_elapsed_values = [
        float(row["elapsed_s"]) for row in selected_fortran_rows if row.get("elapsed_s") is not None
    ]
    selected_fortran_memory_values = [
        float(row["memory_mb"]) for row in selected_fortran_rows if row.get("memory_mb") is not None
    ]
    cached_points = sum(1 for row in ok_rows if row.get("cached"))
    newly_run_points = sum(1 for row in ok_rows if (not row.get("cached")) and row.get("elapsed_s") is not None)
    failed_points = len([row for row in sfincs_rows if row["status"] not in {"ok", "missing"}])
    if convergence is not None:
        errorbar = np.asarray(convergence.get("jdotb_si_errorbar", []), dtype=float)
        rel_errorbar = np.asarray(convergence.get("jdotb_si_errorbar_rel_to_baseline", []), dtype=float)
        finite_errorbar = errorbar[np.isfinite(errorbar)]
        finite_rel_errorbar = rel_errorbar[np.isfinite(rel_errorbar)]
        max_errorbar = None if finite_errorbar.size == 0 else float(np.max(finite_errorbar))
        max_rel_errorbar = None if finite_rel_errorbar.size == 0 else float(np.max(finite_rel_errorbar))
    else:
        max_errorbar = None
        max_rel_errorbar = None
    archived_surfaces = _available_surface_values(case_root)
    whole_radius_scan = len(s_values) == len(archived_surfaces) and np.allclose(s_values, archived_surfaces)
    scan_kind = "whole-radius" if whole_radius_scan else f"{len(s_values)}-surface"
    if args.hide_fortran:
        resolution_status = "JAX-vs-Redl"
    else:
        resolution_status = "same-resolution" if same_resolution_fortran else "mixed-grid"
    payload = {
        "workflow": (
            "qs_paper_sfincs_jax_redl_bootstrap_current"
            if args.hide_fortran
            else "qs_paper_sfincs_jax_fortran_v3_redl_bootstrap_current"
        ),
        "paper": {
            "arxiv": "https://arxiv.org/abs/2205.02914",
            "zenodo_root": str(zenodo_root),
        },
        "case": case.key,
        "case_title": case.title,
        "plot_title": f"{case.title} ({scan_kind} {resolution_status} diagnostic)",
        "claim_boundary": (
            (
                f"{scan_kind} SFINCS_JAX-vs-Redl diagnostic; no SFINCS Fortran v3 "
                "executable or archived Fortran sfincsOutput.h5 overlay is used."
            )
            if args.hide_fortran
            else (
                f"{scan_kind} same-resolution SFINCS_JAX/SFINCS Fortran v3 diagnostic; "
                "production parity still requires residual, convergence, and accuracy gates."
            )
            if same_resolution_fortran
            else (
                f"Fast {scan_kind} scan for workflow timing and radial-shape diagnostics; "
                "not a SFINCS_JAX/SFINCS Fortran v3 same-resolution parity claim because the plotted grids differ."
            )
        ),
        "hide_fortran": bool(args.hide_fortran),
        "fortran_case_root": None if args.hide_fortran else str(fortran_case_root),
        "wout_path": str(wout_path),
        "sfincs_resolution_label": f"{args.ntheta}x{args.nzeta}x{args.nxi}x{args.nx}",
        "sfincs_resolution": {
            "Ntheta": int(args.ntheta),
            "Nzeta": int(args.nzeta),
            "Nxi": int(args.nxi),
            "Nx": int(args.nx),
            "solverTolerance": float(args.solver_tolerance),
            "solveMethod": None if args.solve_method is None else str(args.solve_method),
        },
        "resolution_comparison": resolution_comparison,
        "fortran_errorbars": {
            "source": None if args.fortran_errorbar_json is None else str(args.fortran_errorbar_json),
            "available_points": int(len(fortran_errorbars)),
            "definition": (
                "Optional SFINCS Fortran v3 error bars are loaded from an explicit sidecar. "
                "When absent, no Fortran uncertainty is plotted or inferred from a single archived run."
            ),
        },
        "redl": {
            "s": redl_s.tolist(),
            "jdotb_si": redl_curve.tolist(),
            "profiles": {
                "ne_m^-3": "4.13e20 * (1 - s^5)",
                "Te_eV": "12.0e3 * (1 - s)",
                "Ti_eV": "12.0e3 * (1 - s)",
                "Zeff": "1",
            },
        },
        "sfincs_fortran_v3": fortran_rows,
        "sfincs_jax": sfincs_rows,
        "performance": {
            "runtime_total_s": {
                "sfincs_jax": float(sum(elapsed_values)) if elapsed_values else None,
                "sfincs_fortran_v3": (
                    float(sum(selected_fortran_elapsed_values)) if selected_fortran_elapsed_values else None
                ),
                "sfincs_jax_points": int(len(elapsed_values)),
                "sfincs_fortran_v3_points": int(len(selected_fortran_elapsed_values)),
                "definition": "Sum of per-surface solve wall times for the radial points shown in the figure.",
            },
            "memory_peak_mb": {
                "sfincs_jax": float(max(jax_memory_values)) if jax_memory_values else None,
                "sfincs_fortran_v3": (
                    float(max(selected_fortran_memory_values)) if selected_fortran_memory_values else None
                ),
                "sfincs_jax_points": int(len(jax_memory_values)),
                "sfincs_fortran_v3_points": int(len(selected_fortran_memory_values)),
                "definition": (
                    "Peak over plotted surfaces. JAX uses linear-solver factor plus CSR estimates from HDF5; "
                    "Fortran v3 uses archived MUMPS effective total factor memory when present."
                ),
            },
            "memory_definition": (
                "JAX: factor+CSR estimate."
                if args.hide_fortran
                else "JAX: factor+CSR estimate. Fortran: parsed MUMPS memory metric or process RSS from logs."
            ),
        },
        "metrics": {
            "requested_points": len(sfincs_rows),
            "completed_points": len(ok_rows),
            "cached_points": int(cached_points),
            "newly_run_points": int(newly_run_points),
            "timed_points": int(len(elapsed_values)),
            "failed_points": int(failed_points),
            "fortran_points": len(ok_fortran_rows),
            "sfincs_jax_elapsed_s_sum": float(sum(elapsed_values)) if elapsed_values else None,
            "sfincs_jax_elapsed_s_mean": float(np.mean(elapsed_values)) if elapsed_values else None,
            "sfincs_jax_elapsed_s_min": float(np.min(elapsed_values)) if elapsed_values else None,
            "sfincs_jax_elapsed_s_max": float(np.max(elapsed_values)) if elapsed_values else None,
            "sfincs_jax_solver_memory_mb_max": float(np.max(jax_memory_values)) if jax_memory_values else None,
            "sfincs_fortran_v3_elapsed_s_sum_on_jax_surfaces": (
                float(sum(selected_fortran_elapsed_values)) if selected_fortran_elapsed_values else None
            ),
            "sfincs_fortran_v3_mumps_memory_mb_max_on_jax_surfaces": (
                float(max(selected_fortran_memory_values)) if selected_fortran_memory_values else None
            ),
            "max_relative_difference": max_jax_redl,
            "max_jax_relative_difference_vs_redl": max_jax_redl,
            "max_jax_relative_difference_vs_fortran": max_jax_fortran,
            "max_jax_relative_difference_vs_fortran_same_resolution": (
                max_jax_fortran if same_resolution_fortran else None
            ),
            "max_jax_relative_difference_vs_fortran_mixed_resolution": (
                None if same_resolution_fortran else max_jax_fortran
            ),
            "max_fortran_relative_difference_vs_redl": max_fortran_redl,
            "max_fortran_relative_difference_vs_redl_on_jax_surfaces": max_selected_fortran_redl,
            "max_errorbar_jdotb_si": max_errorbar,
            "max_errorbar_rel_to_baseline": max_rel_errorbar,
        },
    }
    if convergence is not None:
        payload["convergence_errorbars"] = convergence

    _write_plot_artifacts(payload, fig_dir=args.fig_dir, stem=args.stem, print_json=args.json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
