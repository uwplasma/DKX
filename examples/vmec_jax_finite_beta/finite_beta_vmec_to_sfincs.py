#!/usr/bin/env python
"""Finite-beta VMEC-JAX to sfincs_jax bootstrap-current example.

This script is intentionally a single runnable example:

1. Read a finite-beta VMEC input deck.
2. Run VMEC-JAX and write a VMEC-style ``wout`` file.
3. Use that ``wout`` directly in a geometryScheme=5 sfincs_jax input.
4. Scan normalized radial electric field ``Er`` on several flux surfaces.
5. Build radial profiles versus normalized toroidal flux ``psi_N = r_N^2`` for
   the ambipolar radial electric field and bootstrap current, marking roots when
   the bounded scan brackets them.
6. Run a bounded convergence check with tighter ambipolar-root brackets and
   annotate the figure with the profile differences and pass/fail status.

The default grid is intended for the checked documentation figure, not for the
fastest possible smoke test. Use ``--quick`` or reduce the SFINCS resolution
flags for quick local tests; increase ``--vmec-max-iter``, ``--r-n-values``,
``--er-values``, and the resolution flags for production-quality studies.

This is a primal finite-beta transport workflow, not an autodiff example.
``vmec_jax`` is used to generate VMEC-style equilibrium data, but this script
hands a ``wout`` file to ``sfincs_jax`` and runs non-differentiated kinetic
solves.  Kinetic-gradient work is intentionally deferred to separate lanes with
explicit gradient gates.

Run from the repository root:

    python examples/vmec_jax_finite_beta/finite_beta_vmec_to_sfincs.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - dependency error path
    raise SystemExit("This example requires matplotlib. Install sfincs_jax normally first.") from exc

from sfincs_jax.ambipolar import radial_current_from_output  # noqa: E402
from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5  # noqa: E402


DEFAULT_INPUT = Path(__file__).with_name("input.nfp2_QA_finite_beta")
DEFAULT_OUT_DIR = Path(__file__).with_name("output")
DEFAULT_R_N_VALUES = "0.15,0.30,0.50,0.70,0.85"
DEFAULT_ER_VALUES = "-9,-7,-5,-3,-1"


@dataclass(frozen=True)
class RunRecord:
    r_n: float
    er: float
    radial_current: float
    bootstrap_current: float
    ion_particle_flux_rhat: float
    electron_particle_flux_rhat: float
    ion_heat_flux_rhat: float
    electron_heat_flux_rhat: float
    output_h5: str


@dataclass(frozen=True)
class SurfaceProfileRecord:
    r_n: float
    psi_n: float
    roots_er: list[float]
    bootstrap_current_at_roots: list[float]
    selected_ambipolar_er: float | None
    selected_bootstrap_current: float | None
    scan_dir: str


def _psi_n_from_r_n(r_n: float) -> float:
    """Return normalized toroidal flux for the SFINCS ``r_N`` radial label.

    SFINCS convention 3 uses ``r_N``, proportional to the square root of the
    normalized toroidal flux. The radial profile panels use ``psi_N = r_N^2`` on
    the x-axis so the plotted radius is the normalized toroidal flux itself.
    """

    return float(r_n) * float(r_n)


def _setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "lines.linewidth": 2.0,
        }
    )


def _format_er_dir(value: float) -> str:
    # Match the short, readable style used by sfincsScan_2-like outputs.
    text = f"{float(value):.4g}".replace("-", "m").replace("+", "")
    return f"Er{text}"


def _format_r_dir(value: float) -> str:
    text = f"{float(value):.4g}".replace("-", "m").replace("+", "").replace(".", "p")
    return f"rN{text}"


def _resolution_tag(*, ntheta: int, nzeta: int, nxi: int, nl: int, nx: int) -> str:
    return f"Nt{int(ntheta)}_Nz{int(nzeta)}_Nxi{int(nxi)}_NL{int(nl)}_Nx{int(nx)}"


def _parse_er_values(text: str) -> list[float]:
    values = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if len(values) < 2:
        raise ValueError("--er-values must contain at least two comma-separated values.")
    return values


def _parse_r_values(text: str) -> list[float]:
    values = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if not values:
        raise ValueError("--r-n-values must contain at least one comma-separated value.")
    for value in values:
        if not (0.0 < value < 1.0):
            raise ValueError("All --r-n-values entries must satisfy 0 < r_N < 1.")
    return _unique_sorted(values)


def _unique_sorted(values: list[float], *, atol: float = 1.0e-10) -> list[float]:
    out: list[float] = []
    for value in sorted(float(v) for v in values):
        if not any(abs(value - old) <= atol for old in out):
            out.append(value)
    return out


def _ensure_surface(values: list[float], r_n: float) -> list[float]:
    if not (0.0 < float(r_n) < 1.0):
        raise ValueError("--r-n must satisfy 0 < r_N < 1.")
    return _unique_sorted([*values, float(r_n)])


def _require_vmec_jax():
    candidates: list[Path] = []
    env_path = os.environ.get("SFINCS_JAX_VMEC_JAX_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            _REPO_ROOT.parent / "vmec_jax",
            _REPO_ROOT.parent.parent / "vmec_jax",
            Path.home() / "local" / "vmec_jax",
        ]
    )
    for candidate in candidates:
        package_init = candidate / "vmec_jax" / "__init__.py"
        if package_init.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break
    try:
        import vmec_jax as vj
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "This example requires vmec_jax. Install it with `pip install vmec-jax` "
            "or `pip install -e /path/to/vmec_jax`."
        ) from exc
    return vj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_scalar_from_wout(path: Path, name: str) -> float | None:
    try:
        from netCDF4 import Dataset
    except ImportError:  # pragma: no cover - dependency is required by sfincs_jax
        return None
    try:
        with Dataset(path) as ds:
            if name not in ds.variables:
                return None
            return float(np.asarray(ds.variables[name][...]).reshape(()))
    except Exception:
        return None


def _summarize_existing_wout(path: Path, *, vmec_input: Path, vmec_max_iter: int) -> dict[str, Any]:
    """Build the figure metadata from an existing VMEC ``wout`` file."""

    def scalar(name: str, default: float = np.nan) -> float:
        value = _read_scalar_from_wout(path, name)
        return float(default if value is None else value)

    summary = {
        "input": str(vmec_input.resolve()),
        "wout": str(path.resolve()),
        "elapsed_s": 0.0,
        "vmec_max_iter": int(vmec_max_iter),
        "reused_existing_wout": True,
        "nfp": int(scalar("nfp", 0.0)),
        "mpol": int(scalar("mpol", 0.0)),
        "ntor": int(scalar("ntor", 0.0)),
        "fsqr": scalar("fsqr"),
        "fsqz": scalar("fsqz"),
        "fsql": scalar("fsql"),
        "normalization_scalars": {
            "Rmajor_p": scalar("Rmajor_p"),
            "Aminor_p": scalar("Aminor_p"),
            "aspect": scalar("aspect"),
        },
    }
    summary["fsq_total"] = float(summary["fsqr"] + summary["fsqz"] + summary["fsql"])
    return summary


def _fill_wout_scalar_metadata(path: Path, *, run: Any, vj: Any) -> dict[str, float]:
    """Fill VMEC scalar metadata required by sfincs_jax if VMEC-JAX left it zero.

    Some VMEC-JAX development wout writers can emit valid mode/profile data with
    zero ``Aminor_p``/``Rmajor_p`` summary scalars. sfincs_jax uses these scalars
    only for normalization. We compute them from the solved boundary state and
    patch the NetCDF file in-place so the file is self-contained for users and
    for downstream tools.
    """

    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - dependency is required by sfincs_jax
        raise SystemExit("This example requires netCDF4, which is a normal sfincs_jax dependency.") from exc

    aspect = float(vj.equilibrium_aspect_ratio_from_state(state=run.state, static=run.static))
    rcos = np.asarray(run.state.Rcos, dtype=np.float64)
    if rcos.ndim != 2 or rcos.shape[0] < 1 or rcos.shape[1] < 1:
        raise RuntimeError("Could not infer Rmajor_p from VMEC-JAX state.Rcos.")
    rmajor = float(rcos[-1, 0])
    if not np.isfinite(rmajor) or rmajor <= 0.0:
        raise RuntimeError(f"Inferred invalid Rmajor_p={rmajor}.")
    if not np.isfinite(aspect) or aspect <= 0.0:
        raise RuntimeError(f"Inferred invalid aspect={aspect}.")
    aminor = float(rmajor / aspect)

    replacements = {
        "Rmajor_p": rmajor,
        "Aminor_p": aminor,
        "aspect": aspect,
    }
    with Dataset(path, "r+") as ds:
        for name, value in replacements.items():
            if name in ds.variables:
                current = float(np.asarray(ds.variables[name][...]).reshape(()))
                if (not np.isfinite(current)) or current <= 0.0:
                    ds.variables[name][...] = value
            else:
                var = ds.createVariable(name, "f8")
                var[...] = value

    return {
        "Rmajor_p": float(_read_scalar_from_wout(path, "Rmajor_p") or rmajor),
        "Aminor_p": float(_read_scalar_from_wout(path, "Aminor_p") or aminor),
        "aspect": float(_read_scalar_from_wout(path, "aspect") or aspect),
    }


def run_vmec_jax_to_wout(
    *,
    vmec_input: Path,
    out_dir: Path,
    vmec_max_iter: int,
    skip_existing: bool,
    verbose: bool,
) -> tuple[Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    wout_path = out_dir / "wout_nfp2_QA_finite_beta_vmec_jax.nc"
    if bool(skip_existing) and wout_path.exists():
        print(f"  VMEC-JAX wout: reusing {wout_path}", flush=True)
        return wout_path, _summarize_existing_wout(wout_path, vmec_input=vmec_input, vmec_max_iter=vmec_max_iter)

    vj = _require_vmec_jax()
    t0 = time.perf_counter()
    run = vj.run_fixed_boundary(
        vmec_input,
        max_iter=int(vmec_max_iter),
        use_initial_guess=False,
        vmec_project=False,
        multigrid=False,
        verbose=bool(verbose),
    )
    wout = vj.write_wout_from_fixed_boundary_run(
        wout_path,
        run,
        include_fsq=True,
        fast_bcovar=True,
    )
    scalars = _fill_wout_scalar_metadata(wout_path, run=run, vj=vj)
    elapsed = float(time.perf_counter() - t0)

    fsqr = float(getattr(wout, "fsqr", np.nan))
    fsqz = float(getattr(wout, "fsqz", np.nan))
    fsql = float(getattr(wout, "fsql", np.nan))
    summary = {
        "input": str(vmec_input.resolve()),
        "wout": str(wout_path.resolve()),
        "elapsed_s": elapsed,
        "vmec_max_iter": int(vmec_max_iter),
        "nfp": int(getattr(wout, "nfp", 0)),
        "mpol": int(getattr(wout, "mpol", 0)),
        "ntor": int(getattr(wout, "ntor", 0)),
        "fsqr": fsqr,
        "fsqz": fsqz,
        "fsql": fsql,
        "fsq_total": float(fsqr + fsqz + fsql),
        "normalization_scalars": scalars,
    }
    return wout_path, summary


def _sfincs_template(
    *,
    wout_path: Path,
    er: float,
    r_n: float,
    ntheta: int,
    nzeta: int,
    nxi: int,
    nl: int,
    nx: int,
    solver_tolerance: float,
    nu_n: float,
) -> str:
    return f"""! Generated by examples/vmec_jax_finite_beta/finite_beta_vmec_to_sfincs.py
&general
  RHSMode = 1
  saveMatricesAndVectorsInBinary = .false.
/

&geometryParameters
  geometryScheme = 5
  equilibriumFile = "{wout_path}"
  inputRadialCoordinate = 3
  inputRadialCoordinateForGradients = 4
  rN_wish = {float(r_n):.16g}
  VMECRadialOption = 0
/

&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.45509d-4
  nHats = 1.0d+0 1.0d+0
  THats = 1.0d+0 1.0d+0
  dNHatdrHats = -1.0d+0 -1.0d+0
  dTHatdrHats = -2.0d+0 -2.0d+0
/

&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = {float(nu_n):.16g}
  Er = {float(er):.16g}
  collisionOperator = 0
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
  useDKESExBDrift = .false.
  includePhi1 = .false.
/

&resolutionParameters
  Ntheta = {int(ntheta)}
  Nzeta = {int(nzeta)}
  Nxi = {int(nxi)}
  NL = {int(nl)}
  Nx = {int(nx)}
  solverTolerance = {float(solver_tolerance):.16g}
/

&otherNumericalParameters
  Nxi_for_x_option = 0
  xGridScheme = 5
/

&preconditionerOptions
/

&export_f
  export_full_f = .false.
  export_delta_f = .false.
/
"""


def _last_species_values(data: dict[str, Any], key: str, n_species: int) -> np.ndarray:
    arr = np.asarray(data[key], dtype=np.float64)
    if arr.ndim == 0:
        return np.repeat(float(arr), n_species)
    if arr.ndim == 1:
        if arr.size == n_species:
            return arr
        return np.repeat(float(arr.reshape(-1)[-1]), n_species)
    if arr.ndim == 2:
        return arr[:, -1]
    raise ValueError(f"Unexpected rank for {key}: {arr.ndim}")


def _bootstrap_current(data: dict[str, Any]) -> float:
    arr = np.asarray(data["FSABjHat"], dtype=np.float64)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[-1])


def _record_from_output(*, r_n: float, er: float, output_h5: Path) -> RunRecord:
    data = read_sfincs_h5(output_h5)
    n_species = int(np.asarray(data["Nspecies"]).reshape(()))
    particle = _last_species_values(data, "particleFlux_vm_rHat", n_species)
    heat = _last_species_values(data, "heatFlux_vm_rHat", n_species)
    if n_species < 2:
        particle = np.pad(particle, (0, 2 - n_species), mode="edge")
        heat = np.pad(heat, (0, 2 - n_species), mode="edge")
    return RunRecord(
        r_n=float(r_n),
        er=float(er),
        radial_current=float(radial_current_from_output(data)),
        bootstrap_current=_bootstrap_current(data),
        ion_particle_flux_rhat=float(particle[0]),
        electron_particle_flux_rhat=float(particle[1]),
        ion_heat_flux_rhat=float(heat[0]),
        electron_heat_flux_rhat=float(heat[1]),
        output_h5=str(output_h5.resolve()),
    )


def _run_or_load_sfincs_record(
    *,
    wout_path: Path,
    scan_dir: Path,
    er: float,
    r_n: float,
    ntheta: int,
    nzeta: int,
    nxi: int,
    nl: int,
    nx: int,
    solver_tolerance: float,
    nu_n: float,
    skip_existing: bool,
    verbose: bool,
    label: str,
) -> RunRecord:
    run_dir = scan_dir / _format_er_dir(er)
    run_dir.mkdir(parents=True, exist_ok=True)
    input_path = run_dir / "input.namelist"
    output_h5 = run_dir / "sfincsOutput.h5"
    input_path.write_text(
        _sfincs_template(
            wout_path=wout_path.resolve(),
            er=er,
            r_n=r_n,
            ntheta=ntheta,
            nzeta=nzeta,
            nxi=nxi,
            nl=nl,
            nx=nx,
            solver_tolerance=solver_tolerance,
            nu_n=nu_n,
        ),
        encoding="utf-8",
    )
    if skip_existing and output_h5.exists():
        print(f"{label} r_N={r_n:g}, Er={er:g}: reusing {output_h5}", flush=True)
    else:
        t0 = time.perf_counter()
        print(f"{label} r_N={r_n:g}, Er={er:g}: running sfincs_jax", flush=True)
        write_sfincs_jax_output_h5(
            input_namelist=input_path,
            output_path=output_h5,
            compute_solution=True,
            overwrite=True,
            verbose=bool(verbose),
            differentiable=False,
        )
        print(f"{label} r_N={r_n:g}, Er={er:g}: done in {time.perf_counter() - t0:.2f} s", flush=True)
    return _record_from_output(r_n=r_n, er=er, output_h5=output_h5)


def _sign_change_intervals(records: list[RunRecord]) -> list[tuple[RunRecord, RunRecord]]:
    ordered = sorted(records, key=lambda record: record.er)
    intervals: list[tuple[RunRecord, RunRecord]] = []
    for left, right in zip(ordered[:-1], ordered[1:], strict=False):
        y0 = float(left.radial_current)
        y1 = float(right.radial_current)
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 == 0.0 or y1 == 0.0 or y0 * y1 < 0.0:
            intervals.append((left, right))
    return intervals


def _refine_ambipolar_brackets(
    *,
    records: list[RunRecord],
    wout_path: Path,
    scan_dir: Path,
    r_n: float,
    ntheta: int,
    nzeta: int,
    nxi: int,
    nl: int,
    nx: int,
    solver_tolerance: float,
    nu_n: float,
    skip_existing: bool,
    verbose: bool,
    target_width: float,
    max_iterations: int,
) -> list[RunRecord]:
    """Add midpoint solves until each bracketed ambipolar root has enough Er resolution."""

    if target_width <= 0.0 or max_iterations <= 0:
        return sorted(records, key=lambda record: record.er)

    known_by_er = {round(float(record.er), 12): record for record in records}
    for iteration in range(1, int(max_iterations) + 1):
        intervals = _sign_change_intervals(list(known_by_er.values()))
        pending: list[float] = []
        for left, right in intervals:
            width = abs(float(right.er) - float(left.er))
            if width <= float(target_width):
                continue
            midpoint = 0.5 * (float(left.er) + float(right.er))
            key = round(midpoint, 12)
            if key not in known_by_er:
                pending.append(midpoint)
        if not pending:
            break
        for refine_index, er in enumerate(sorted(pending), start=1):
            label = f"[root-refine {iteration}.{refine_index}/{len(pending)}]"
            record = _run_or_load_sfincs_record(
                wout_path=wout_path,
                scan_dir=scan_dir,
                er=er,
                r_n=r_n,
                ntheta=ntheta,
                nzeta=nzeta,
                nxi=nxi,
                nl=nl,
                nx=nx,
                solver_tolerance=solver_tolerance,
                nu_n=nu_n,
                skip_existing=skip_existing,
                verbose=verbose,
                label=label,
            )
            known_by_er[round(float(record.er), 12)] = record
    return sorted(known_by_er.values(), key=lambda record: record.er)


def run_sfincs_er_scan(
    *,
    wout_path: Path,
    out_dir: Path,
    er_values: list[float],
    r_n: float,
    ntheta: int,
    nzeta: int,
    nxi: int,
    nl: int,
    nx: int,
    solver_tolerance: float,
    nu_n: float,
    skip_existing: bool,
    verbose: bool,
    root_refine_width: float,
    root_refine_max_iter: int,
    scan_dir: Path | None = None,
) -> tuple[list[RunRecord], Path]:
    scan_dir = scan_dir or (out_dir / "sfincs_er_scan")
    scan_dir.mkdir(parents=True, exist_ok=True)
    records: list[RunRecord] = []
    values = sorted([float(v) for v in er_values])
    scan_t0 = time.perf_counter()

    for index, er in enumerate(values, start=1):
        records.append(
            _run_or_load_sfincs_record(
                wout_path=wout_path,
                scan_dir=scan_dir,
                er=er,
                r_n=r_n,
                ntheta=ntheta,
                nzeta=nzeta,
                nxi=nxi,
                nl=nl,
                nx=nx,
                solver_tolerance=solver_tolerance,
                nu_n=nu_n,
                skip_existing=skip_existing,
                verbose=verbose,
                label=f"[{index}/{len(values)}]",
            )
        )
    records = _refine_ambipolar_brackets(
        records=records,
        wout_path=wout_path,
        scan_dir=scan_dir,
        r_n=r_n,
        ntheta=ntheta,
        nzeta=nzeta,
        nxi=nxi,
        nl=nl,
        nx=nx,
        solver_tolerance=solver_tolerance,
        nu_n=nu_n,
        skip_existing=skip_existing,
        verbose=verbose,
        target_width=float(root_refine_width),
        max_iterations=int(root_refine_max_iter),
    )

    print(f"SFINCS Er scan at r_N={r_n:g} completed in {time.perf_counter() - scan_t0:.2f} s")
    return records, scan_dir


def _ambipolar_roots(records: list[RunRecord]) -> list[float]:
    from scipy.interpolate import PchipInterpolator, interp1d
    from scipy.optimize import brentq

    x = np.asarray([r.er for r in records], dtype=np.float64)
    y = np.asarray([r.radial_current for r in records], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x.size < 2 or not (np.nanmin(y) <= 0.0 <= np.nanmax(y)):
        return []
    if x.size >= 3:
        try:
            interp = PchipInterpolator(x, y)
        except Exception:
            interp = interp1d(x, y, kind="linear")
    else:
        interp = interp1d(x, y, kind="linear")

    roots: list[float] = []
    for i in range(x.size - 1):
        y0 = float(y[i])
        y1 = float(y[i + 1])
        if y0 == 0.0:
            roots.append(float(x[i]))
        if y0 * y1 < 0.0:
            roots.append(float(brentq(lambda val: float(interp(val)), float(x[i]), float(x[i + 1]))))
    if float(y[-1]) == 0.0:
        roots.append(float(x[-1]))
    deduped: list[float] = []
    for root in roots:
        if not any(abs(root - old) < 1.0e-8 for old in deduped):
            deduped.append(root)
    return deduped


def _root_interpolated_values(records: list[RunRecord], roots: list[float], key: str) -> list[float]:
    if not roots:
        return []
    from scipy.interpolate import PchipInterpolator, interp1d

    x = np.asarray([r.er for r in records], dtype=np.float64)
    y = np.asarray([getattr(r, key) for r in records], dtype=np.float64)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if x.size >= 3:
        try:
            interp = PchipInterpolator(x, y)
        except Exception:
            interp = interp1d(x, y, kind="linear")
    else:
        interp = interp1d(x, y, kind="linear")
    return [float(interp(root)) for root in roots]


def _selected_root_index(
    roots: list[float],
    *,
    previous_root: float | None,
    preferred_er: float,
) -> int | None:
    if not roots:
        return None
    target = float(preferred_er) if previous_root is None else float(previous_root)
    return int(np.argmin(np.abs(np.asarray(roots, dtype=np.float64) - target)))


def _finite_beta_workflow_contract() -> dict[str, Any]:
    return {
        "workflow": "finite_beta_vmec_jax_to_sfincs_jax_bootstrap_er_profile",
        "contract_version": 1,
        "differentiability": {
            "vmec_jax_fixed_boundary_run": "primal_setup_not_differentiated_by_this_example",
            "vmec_wout_file_handoff": "file_handoff_not_differentiated",
            "sfincs_geometry_scheme5": "primal_geometry_evaluation_not_differentiated",
            "sfincs_kinetic_transport_solve": "primal_solve_only_not_differentiated",
            "radial_profile_postprocessing": "postprocessing_not_differentiated",
        },
        "no_overclaim_gate": {
            "status": "pass",
            "claim_scope": "finite_beta_primal_profile_with_convergence_audit",
            "full_transport_gradients_claimed": False,
            "kinetic_gradient_status": "deferred_not_covered_by_this_example",
        },
    }


def _radial_profile_provenance(
    *,
    profile: list[SurfaceProfileRecord],
    convergence_profile: list[SurfaceProfileRecord] | None,
    accuracy: dict[str, float | int | bool | None],
    representative_r_n: float,
) -> dict[str, Any]:
    surfaces = [
        {
            "r_n": float(row.r_n),
            "psi_n": float(row.psi_n),
            "psi_n_formula": "psi_N = r_N^2",
            "roots_recorded": len(row.roots_er),
            "selected_ambipolar_er_recorded": row.selected_ambipolar_er is not None,
            "selected_bootstrap_current_recorded": row.selected_bootstrap_current is not None,
            "scan_dir": str(row.scan_dir),
        }
        for row in profile
    ]
    convergence_surfaces = [
        {
            "r_n": float(row.r_n),
            "psi_n": float(row.psi_n),
            "roots_recorded": len(row.roots_er),
            "scan_dir": str(row.scan_dir),
        }
        for row in (convergence_profile or [])
    ]
    return {
        "radial_coordinate_input": "r_N",
        "radial_profile_axis": "normalized toroidal flux psi_N",
        "psi_n_formula": "psi_N = r_N^2",
        "representative_r_n": float(representative_r_n),
        "branch_selection": (
            "At the innermost requested surface, choose the root nearest "
            "--profile-root-preference; on later surfaces, choose the root nearest "
            "the previously selected root."
        ),
        "all_bracketed_roots_preserved": all(
            len(row.roots_er) == len(row.bootstrap_current_at_roots) for row in profile
        ),
        "surfaces": surfaces,
        "convergence": {
            "profile_present": bool(convergence_profile),
            "surfaces": convergence_surfaces,
            "surfaces_checked": int(accuracy.get("surfaces_checked") or 0),
            "passed": bool(accuracy.get("passed", False)),
            "max_abs_er": accuracy.get("max_abs_er"),
            "max_abs_bootstrap": accuracy.get("max_abs_bootstrap"),
        },
    }


def build_radial_profile(
    *,
    scans_by_radius: dict[float, tuple[list[RunRecord], Path]],
    preferred_er: float,
) -> list[SurfaceProfileRecord]:
    """Select a continuous ambipolar branch across the radial scan.

    A finite Er scan can bracket multiple ambipolar roots. For a compact example
    profile we choose the root nearest ``preferred_er`` on the first surface, then
    continue with the nearest root to the previous selected value. The summary
    still records every bracketed root so users can inspect branch ambiguity.
    """

    profile: list[SurfaceProfileRecord] = []
    previous_root: float | None = None
    for r_n in sorted(scans_by_radius):
        records, scan_dir = scans_by_radius[r_n]
        roots = _ambipolar_roots(records)
        bootstrap_at_roots = _root_interpolated_values(records, roots, "bootstrap_current")
        selected_index = _selected_root_index(roots, previous_root=previous_root, preferred_er=preferred_er)
        if selected_index is None:
            selected_er = None
            selected_bootstrap = None
        else:
            selected_er = float(roots[selected_index])
            selected_bootstrap = float(bootstrap_at_roots[selected_index])
            previous_root = selected_er
        profile.append(
            SurfaceProfileRecord(
                r_n=float(r_n),
                psi_n=_psi_n_from_r_n(float(r_n)),
                roots_er=[float(v) for v in roots],
                bootstrap_current_at_roots=[float(v) for v in bootstrap_at_roots],
                selected_ambipolar_er=selected_er,
                selected_bootstrap_current=selected_bootstrap,
                scan_dir=str(scan_dir.resolve()),
            )
        )
    return profile


def run_sfincs_radial_er_scan(
    *,
    wout_path: Path,
    out_dir: Path,
    er_values: list[float],
    r_n_values: list[float],
    ntheta: int,
    nzeta: int,
    nxi: int,
    nl: int,
    nx: int,
    solver_tolerance: float,
    nu_n: float,
    skip_existing: bool,
    verbose: bool,
    preferred_er: float,
    root_refine_width: float,
    root_refine_max_iter: int,
    scan_root: Path | None = None,
) -> tuple[dict[float, tuple[list[RunRecord], Path]], list[SurfaceProfileRecord], Path]:
    scan_root = scan_root or (
        out_dir
        / f"sfincs_radial_er_scan_{_resolution_tag(ntheta=ntheta, nzeta=nzeta, nxi=nxi, nl=nl, nx=nx)}"
    )
    scan_root.mkdir(parents=True, exist_ok=True)
    scans_by_radius: dict[float, tuple[list[RunRecord], Path]] = {}
    t0 = time.perf_counter()
    surfaces = _unique_sorted([float(v) for v in r_n_values])
    for surface_index, r_n in enumerate(surfaces, start=1):
        print(f"Surface {surface_index}/{len(surfaces)}: scanning r_N={r_n:g}")
        records, scan_dir = run_sfincs_er_scan(
            wout_path=wout_path,
            out_dir=out_dir,
            er_values=er_values,
            r_n=r_n,
            ntheta=ntheta,
            nzeta=nzeta,
            nxi=nxi,
            nl=nl,
            nx=nx,
            solver_tolerance=solver_tolerance,
            nu_n=nu_n,
            skip_existing=skip_existing,
            verbose=verbose,
            root_refine_width=root_refine_width,
            root_refine_max_iter=root_refine_max_iter,
            scan_dir=scan_root / _format_r_dir(r_n),
        )
        scans_by_radius[float(r_n)] = (records, scan_dir)
    profile = build_radial_profile(scans_by_radius=scans_by_radius, preferred_er=preferred_er)
    print(f"Radial Er scan completed in {time.perf_counter() - t0:.2f} s")
    return scans_by_radius, profile, scan_root


def _nearest_surface_records(
    scans_by_radius: dict[float, tuple[list[RunRecord], Path]],
    r_n: float,
) -> tuple[float, list[RunRecord], Path]:
    radii = np.asarray(sorted(scans_by_radius), dtype=np.float64)
    selected = float(radii[int(np.argmin(np.abs(radii - float(r_n))))])
    records, scan_dir = scans_by_radius[selected]
    return selected, records, scan_dir


def convergence_summary(
    *,
    baseline: list[SurfaceProfileRecord],
    refined: list[SurfaceProfileRecord],
    max_abs_er_tolerance: float,
    max_abs_bootstrap_tolerance: float,
) -> dict[str, float | int | bool | None]:
    baseline_by_r = {round(p.r_n, 12): p for p in baseline}
    refined_by_r = {round(p.r_n, 12): p for p in refined}
    er_diffs: list[float] = []
    er_rel_diffs: list[float] = []
    bootstrap_diffs: list[float] = []
    bootstrap_rel_diffs: list[float] = []
    for key in sorted(set(baseline_by_r).intersection(refined_by_r)):
        coarse = baseline_by_r[key]
        fine = refined_by_r[key]
        if coarse.selected_ambipolar_er is None or fine.selected_ambipolar_er is None:
            continue
        if coarse.selected_bootstrap_current is None or fine.selected_bootstrap_current is None:
            continue
        er_diff = abs(float(coarse.selected_ambipolar_er) - float(fine.selected_ambipolar_er))
        bootstrap_diff = abs(float(coarse.selected_bootstrap_current) - float(fine.selected_bootstrap_current))
        er_diffs.append(er_diff)
        bootstrap_diffs.append(bootstrap_diff)
        er_rel_diffs.append(er_diff / max(abs(float(fine.selected_ambipolar_er)), 1.0))
        bootstrap_rel_diffs.append(bootstrap_diff / max(abs(float(fine.selected_bootstrap_current)), 1.0e-12))
    checked = len(er_diffs)
    max_abs_er = float(np.max(er_diffs)) if checked else None
    max_abs_bootstrap = float(np.max(bootstrap_diffs)) if checked else None
    passed = (
        bool(checked)
        and max_abs_er is not None
        and max_abs_bootstrap is not None
        and max_abs_er <= float(max_abs_er_tolerance)
        and max_abs_bootstrap <= float(max_abs_bootstrap_tolerance)
    )
    return {
        "surfaces_checked": checked,
        "max_abs_er": max_abs_er,
        "max_rel_er": float(np.max(er_rel_diffs)) if checked else None,
        "max_abs_bootstrap": max_abs_bootstrap,
        "max_rel_bootstrap": float(np.max(bootstrap_rel_diffs)) if checked else None,
        "max_abs_er_tolerance": float(max_abs_er_tolerance),
        "max_abs_bootstrap_tolerance": float(max_abs_bootstrap_tolerance),
        "passed": passed,
    }


def _load_reference_output(records: list[RunRecord]) -> dict[str, Any]:
    idx = int(np.argmin(np.abs(np.asarray([r.er for r in records], dtype=np.float64))))
    return read_sfincs_h5(Path(records[idx].output_h5))


def plot_summary(
    *,
    records: list[RunRecord],
    roots: list[float],
    profile: list[SurfaceProfileRecord],
    convergence_profile: list[SurfaceProfileRecord] | None,
    accuracy: dict[str, float | int | bool | None],
    vmec_summary: dict[str, Any],
    out_dir: Path,
    stem: str,
    representative_r_n: float,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    data0 = _load_reference_output(records)
    er = np.asarray([r.er for r in records], dtype=np.float64)
    radial_current = np.asarray([r.radial_current for r in records], dtype=np.float64)
    bootstrap = np.asarray([r.bootstrap_current for r in records], dtype=np.float64)
    ion_flux = np.asarray([r.ion_particle_flux_rhat for r in records], dtype=np.float64)
    electron_flux = np.asarray([r.electron_particle_flux_rhat for r in records], dtype=np.float64)
    b_hat = np.asarray(data0["BHat"], dtype=np.float64)
    if b_hat.ndim > 2:
        b_hat = b_hat[..., 0]

    colors = {
        "radial": "#284b63",
        "bootstrap": "#b24c36",
        "ion": "#2a9d8f",
        "electron": "#6d597a",
        "root": "#1d1d1f",
        "profile": "#d97706",
    }
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.1), constrained_layout=False)

    ax = axes[0, 0]
    profile_x = np.asarray([p.psi_n for p in profile], dtype=np.float64)
    profile_er = np.asarray(
        [np.nan if p.selected_ambipolar_er is None else p.selected_ambipolar_er for p in profile],
        dtype=np.float64,
    )
    labeled_all_roots = False
    for p in profile:
        if p.roots_er:
            ax.plot(
                [p.psi_n] * len(p.roots_er),
                p.roots_er,
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=colors["profile"],
                alpha=0.55,
                label="all roots" if not labeled_all_roots else None,
            )
            labeled_all_roots = True
    mask = np.isfinite(profile_er)
    ax.plot(
        profile_x[mask],
        profile_er[mask],
        "o-",
        color=colors["profile"],
        linewidth=2.6,
        markersize=7.5,
        label=r"selected $E_r(\psi_N)$",
    )
    if convergence_profile:
        convergence_x = np.asarray([p.psi_n for p in convergence_profile], dtype=np.float64)
        convergence_er = np.asarray(
            [np.nan if p.selected_ambipolar_er is None else p.selected_ambipolar_er for p in convergence_profile],
            dtype=np.float64,
        )
        convergence_mask = np.isfinite(convergence_er)
        ax.plot(
            convergence_x[convergence_mask],
            convergence_er[convergence_mask],
            "s--",
            color="black",
            linewidth=1.6,
            markersize=5.8,
            alpha=0.75,
            label="refined scan",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_title(r"Ambipolar electric field vs toroidal flux")
    ax.set_xlabel(r"normalized toroidal flux $\psi_N = r_N^2$")
    ax.set_ylabel(r"ambipolar electric field $E_r(\psi_N)$")
    ax.legend(loc="best")

    ax = axes[0, 1]
    profile_bootstrap = np.asarray(
        [np.nan if p.selected_bootstrap_current is None else p.selected_bootstrap_current for p in profile],
        dtype=np.float64,
    )
    labeled_all_roots = False
    for p in profile:
        if p.bootstrap_current_at_roots:
            ax.plot(
                [p.psi_n] * len(p.bootstrap_current_at_roots),
                p.bootstrap_current_at_roots,
                marker="o",
                linestyle="None",
                markerfacecolor="none",
                markeredgecolor=colors["bootstrap"],
                alpha=0.55,
                label="all roots" if not labeled_all_roots else None,
            )
            labeled_all_roots = True
    mask = np.isfinite(profile_bootstrap)
    ax.plot(
        profile_x[mask],
        profile_bootstrap[mask],
        "o-",
        color=colors["bootstrap"],
        linewidth=2.6,
        markersize=7.5,
        label=r"selected $J_\parallel(\psi_N)$",
    )
    if convergence_profile:
        convergence_x = np.asarray([p.psi_n for p in convergence_profile], dtype=np.float64)
        convergence_bootstrap = np.asarray(
            [
                np.nan if p.selected_bootstrap_current is None else p.selected_bootstrap_current
                for p in convergence_profile
            ],
            dtype=np.float64,
        )
        convergence_mask = np.isfinite(convergence_bootstrap)
        ax.plot(
            convergence_x[convergence_mask],
            convergence_bootstrap[convergence_mask],
            "s--",
            color="black",
            linewidth=1.6,
            markersize=5.8,
            alpha=0.75,
            label="refined scan",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Bootstrap current vs toroidal flux")
    ax.set_xlabel(r"normalized toroidal flux $\psi_N = r_N^2$")
    ax.set_ylabel(r"FSABjHat$(\psi_N)$ at ambipolar $E_r$")
    ax.legend(loc="best")

    ax = axes[0, 2]
    theta_index = np.arange(b_hat.shape[0])
    zeta_index = np.arange(b_hat.shape[1])
    levels = np.linspace(float(np.nanmin(b_hat)), float(np.nanmax(b_hat)), 24)
    im = ax.contourf(zeta_index, theta_index, b_hat, levels=levels, cmap="jet")
    ax.contour(zeta_index, theta_index, b_hat, levels=8, colors="black", linewidths=0.35, alpha=0.35)
    ax.set_title(rf"VMEC-JAX $|\hat B|$ contour at $r_N={representative_r_n:.2f}$")
    ax.set_xlabel("zeta index")
    ax.set_ylabel("theta index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    ax.plot(er, radial_current, "o-", color=colors["radial"], label=r"$j_\psi$")
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_title(rf"Ambipolar scan at $r_N={representative_r_n:.2f}$")
    ax.set_xlabel(r"normalized radial electric field $E_r$")
    ax.set_ylabel(r"radial current $j_\psi$")
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(er, bootstrap, "o-", color=colors["bootstrap"], label=r"$\langle J_\parallel B\rangle$")
    ax.set_title(rf"Bootstrap response at $r_N={representative_r_n:.2f}$")
    ax.set_xlabel(r"normalized radial electric field $E_r$")
    ax.set_ylabel("FSABjHat")
    ax.legend(loc="best")

    ax = axes[1, 2]
    ax.plot(er, ion_flux, "o-", color=colors["ion"], label="ion particle flux")
    ax.plot(er, electron_flux, "s-", color=colors["electron"], label="electron particle flux")
    ax.set_title(rf"Species particle fluxes at $r_N={representative_r_n:.2f}$")
    ax.set_xlabel(r"normalized radial electric field $E_r$")
    ax.set_ylabel(r"$\Gamma_r$")
    ax.legend(loc="best")

    root_bootstrap = _root_interpolated_values(records, roots, "bootstrap_current")
    for root, b_root in zip(roots, root_bootstrap, strict=False):
        axes[1, 0].axvline(root, color=colors["root"], linestyle=":", linewidth=1.4)
        axes[1, 1].axvline(root, color=colors["root"], linestyle=":", linewidth=1.4)
        axes[1, 1].plot([root], [b_root], marker="*", markersize=11, color=colors["root"])

    root_text = ", ".join(f"{root:.4g}" for root in roots) if roots else "not bracketed"
    selected_profile_count = sum(1 for p in profile if p.selected_ambipolar_er is not None)
    fsq_total = float(vmec_summary.get("fsq_total", np.nan))
    normal = dict(vmec_summary.get("normalization_scalars", {}))
    checked = int(accuracy.get("surfaces_checked") or 0)
    max_er = accuracy.get("max_abs_er")
    max_bootstrap = accuracy.get("max_abs_bootstrap")
    passed = bool(accuracy.get("passed", False))
    convergence_text = (
        f"convergence {'PASS' if passed else 'FAIL'} on {checked} surfaces: "
        f"max |Delta Er|={float(max_er):.2g}, max |Delta Jbs|={float(max_bootstrap):.2g}"
        if checked and max_er is not None and max_bootstrap is not None
        else "convergence check skipped"
    )
    subtitle = (
        f"finite-beta nfp=2 QA; roots at r_N={representative_r_n:.2f}: {root_text}; "
        f"profile roots bracketed on {selected_profile_count}/{len(profile)} surfaces; "
        f"{convergence_text}; VMEC fsq_total={fsq_total:.2e}; "
        f"Aminor_p={float(normal.get('Aminor_p', np.nan)):.3g}"
    )
    fig.suptitle("VMEC-JAX to sfincs_jax finite-beta bootstrap-current pipeline", fontsize=14, y=0.985)
    fig.text(0.5, 0.018, subtitle, ha="center", va="bottom", fontsize=9.2)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.91, wspace=0.34, hspace=0.50)

    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png.resolve(), pdf.resolve()


def build_summary(
    *,
    vmec_summary: dict[str, Any],
    records: list[RunRecord],
    roots: list[float],
    profile: list[SurfaceProfileRecord],
    convergence_profile: list[SurfaceProfileRecord] | None,
    accuracy: dict[str, float | int | bool | None],
    figure_png: Path,
    figure_pdf: Path,
    scan_dir: Path,
    representative_r_n: float,
) -> dict[str, Any]:
    return {
        "metadata": {
            "kind": "finite_beta_vmec_jax_to_sfincs_jax_bootstrap_er",
            "source_script": str(Path(__file__).resolve().relative_to(_REPO_ROOT)),
            "scan_dir": str(scan_dir.resolve()),
            "figure_png": str(figure_png),
            "figure_pdf": str(figure_pdf),
            "representative_r_n": float(representative_r_n),
            "normalization": (
                "Er, radial current, bootstrap current, particle fluxes, and heat fluxes are "
                "the normalized quantities written by sfincs_jax."
            ),
            "profile_branch_selection": (
                "At the innermost requested surface, choose the root nearest --profile-root-preference; "
                "on later surfaces, choose the root nearest the previously selected root."
            ),
            "radial_axis": "The radial-profile x-axis is normalized toroidal flux psi_N = r_N^2.",
            "workflow_contract": _finite_beta_workflow_contract(),
            "radial_profile_provenance": _radial_profile_provenance(
                profile=profile,
                convergence_profile=convergence_profile,
                accuracy=accuracy,
                representative_r_n=representative_r_n,
            ),
        },
        "vmec_jax": vmec_summary,
        "sfincs_jax": {
            "runs": [asdict(r) for r in records],
            "ambipolar_roots_er": [float(v) for v in roots],
            "bootstrap_current_at_roots": _root_interpolated_values(records, roots, "bootstrap_current"),
            "radial_current_min": float(np.min([r.radial_current for r in records])),
            "radial_current_max": float(np.max([r.radial_current for r in records])),
            "radial_profile": [asdict(p) for p in profile],
            "convergence_profile": [asdict(p) for p in convergence_profile] if convergence_profile else [],
            "convergence_summary": accuracy,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vmec-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default="finite_beta_vmec_jax_sfincs_bootstrap_er")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use smaller smoke-test settings. The checked documentation figure uses the default validation settings.",
    )
    parser.add_argument(
        "--er-values",
        default=DEFAULT_ER_VALUES,
        help="Comma-separated normalized Er scan values. Defaults bracket the demo radial current.",
    )
    parser.add_argument(
        "--r-n",
        type=float,
        default=0.5,
        help="Representative flux surface shown in the detailed Er-scan panels. Included in --r-n-values.",
    )
    parser.add_argument(
        "--r-n-values",
        default=DEFAULT_R_N_VALUES,
        help=(
            "Comma-separated radial surfaces for the profile scan. Defaults avoid the exact axis and boundary "
            f"while spanning core to edge: {DEFAULT_R_N_VALUES}."
        ),
    )
    parser.add_argument(
        "--profile-root-preference",
        type=float,
        default=0.0,
        help="First-surface ambipolar branch preference when multiple roots are bracketed.",
    )
    parser.add_argument("--vmec-max-iter", type=int, default=5)
    parser.add_argument("--ntheta", type=int, default=7)
    parser.add_argument("--nzeta", type=int, default=7)
    parser.add_argument("--nxi", type=int, default=8)
    parser.add_argument("--nl", type=int, default=6)
    parser.add_argument("--nx", type=int, default=6)
    parser.add_argument(
        "--root-refine-width",
        type=float,
        default=1.25,
        help="Target Er interval width for adaptive midpoint refinement of bracketed ambipolar roots. Use 0 to disable.",
    )
    parser.add_argument(
        "--root-refine-max-iter",
        type=int,
        default=3,
        help="Maximum midpoint-refinement iterations per surface for bracketed ambipolar roots.",
    )
    parser.add_argument("--skip-convergence", action="store_true", help="Skip the root-bracket convergence check.")
    parser.add_argument(
        "--convergence-r-n-values",
        default=DEFAULT_R_N_VALUES,
        help="Comma-separated surfaces for the convergence check. Defaults to the full plotted radial profile.",
    )
    parser.add_argument("--convergence-ntheta", type=int, default=7)
    parser.add_argument("--convergence-nzeta", type=int, default=7)
    parser.add_argument("--convergence-nxi", type=int, default=8)
    parser.add_argument("--convergence-nl", type=int, default=6)
    parser.add_argument("--convergence-nx", type=int, default=6)
    parser.add_argument(
        "--convergence-root-refine-width",
        type=float,
        default=0.625,
        help="Refined Er interval width for the convergence-profile ambipolar-root scan.",
    )
    parser.add_argument(
        "--convergence-root-refine-max-iter",
        type=int,
        default=4,
        help="Maximum midpoint-refinement iterations for the convergence-profile ambipolar roots.",
    )
    parser.add_argument(
        "--convergence-max-abs-er",
        type=float,
        default=0.1,
        help="Absolute tolerance for the ambipolar Er profile comparison.",
    )
    parser.add_argument(
        "--convergence-max-abs-bootstrap",
        type=float,
        default=5.0e-4,
        help="Absolute tolerance for the bootstrap-current profile comparison.",
    )
    parser.add_argument(
        "--require-convergence",
        action="store_true",
        help="Return a non-zero exit status if the convergence profile check fails the requested tolerances.",
    )
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--nu-n", type=float, default=1.0e-2)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--verbose-vmec", action="store_true")
    parser.add_argument("--verbose-sfincs", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    _setup_matplotlib()
    args = _build_parser().parse_args(argv)
    vmec_input = Path(args.vmec_input).resolve()
    if not vmec_input.exists():
        raise FileNotFoundError(vmec_input)
    out_dir = Path(args.out_dir).resolve()
    if bool(args.quick):
        args.er_values = "-30,-20,-10,0,10,20,30"
        args.ntheta = 7
        args.nzeta = 7
        args.nxi = 5
        args.nl = 4
        args.nx = 4
        args.convergence_ntheta = 9
        args.convergence_nzeta = 9
        args.convergence_nxi = 6
        args.convergence_nl = 5
        args.convergence_nx = 5
        args.root_refine_width = 0.0
        args.root_refine_max_iter = 0
        args.convergence_root_refine_width = 0.0
        args.convergence_root_refine_max_iter = 0

    er_values = _parse_er_values(str(args.er_values))
    r_n_values = _ensure_surface(_parse_r_values(str(args.r_n_values)), float(args.r_n))
    convergence_r_n_values = (
        _ensure_surface(_parse_r_values(str(args.convergence_r_n_values)), float(args.r_n))
        if str(args.convergence_r_n_values).strip()
        else r_n_values
    )

    print("Finite-beta VMEC-JAX to sfincs_jax pipeline")
    print(f"  VMEC input: {vmec_input}")
    print(f"  Output dir: {out_dir}")
    print(f"  Er scan: {er_values}")
    print(f"  radial surfaces: {r_n_values}")
    print(
        "  profile grid: "
        f"Ntheta={args.ntheta}, Nzeta={args.nzeta}, "
        f"Nxi={args.nxi}, NL={args.nl}, Nx={args.nx}"
    )
    print(
        "  ambipolar-root refinement: "
        f"target width={float(args.root_refine_width):g}, "
        f"max iterations={int(args.root_refine_max_iter)}"
    )
    if args.skip_convergence:
        print("  convergence check: skipped")
    else:
        print(
            "  convergence grid: "
            f"Ntheta={args.convergence_ntheta}, Nzeta={args.convergence_nzeta}, "
            f"Nxi={args.convergence_nxi}, NL={args.convergence_nl}, Nx={args.convergence_nx}"
        )
        print(
            "  convergence ambipolar-root refinement: "
            f"target width={float(args.convergence_root_refine_width):g}, "
            f"max iterations={int(args.convergence_root_refine_max_iter)}"
        )
        print(
            "  convergence tolerances: "
            f"|Delta Er| <= {float(args.convergence_max_abs_er):g}, "
            f"|Delta Jbs| <= {float(args.convergence_max_abs_bootstrap):g}"
        )

    wout_path, vmec_summary = run_vmec_jax_to_wout(
        vmec_input=vmec_input,
        out_dir=out_dir,
        vmec_max_iter=int(args.vmec_max_iter),
        skip_existing=bool(args.skip_existing),
        verbose=bool(args.verbose_vmec),
    )
    print(
        "  VMEC-JAX wout: "
        f"{wout_path} (fsq_total={float(vmec_summary['fsq_total']):.3e}, "
        f"elapsed={float(vmec_summary['elapsed_s']):.2f}s)"
    )

    scans_by_radius, profile, scan_dir = run_sfincs_radial_er_scan(
        wout_path=wout_path,
        out_dir=out_dir,
        er_values=er_values,
        r_n_values=r_n_values,
        ntheta=int(args.ntheta),
        nzeta=int(args.nzeta),
        nxi=int(args.nxi),
        nl=int(args.nl),
        nx=int(args.nx),
        solver_tolerance=float(args.solver_tolerance),
        nu_n=float(args.nu_n),
        skip_existing=bool(args.skip_existing),
        verbose=bool(args.verbose_sfincs),
        preferred_er=float(args.profile_root_preference),
        root_refine_width=float(args.root_refine_width),
        root_refine_max_iter=int(args.root_refine_max_iter),
    )
    convergence_profile: list[SurfaceProfileRecord] | None = None
    accuracy: dict[str, float | int | bool | None] = {"surfaces_checked": 0, "passed": False}
    if not bool(args.skip_convergence):
        print("Running tighter root-bracket convergence profile check")
        _, convergence_profile, convergence_scan_dir = run_sfincs_radial_er_scan(
            wout_path=wout_path,
            out_dir=out_dir,
            er_values=er_values,
            r_n_values=convergence_r_n_values,
            ntheta=int(args.convergence_ntheta),
            nzeta=int(args.convergence_nzeta),
            nxi=int(args.convergence_nxi),
            nl=int(args.convergence_nl),
            nx=int(args.convergence_nx),
            solver_tolerance=float(args.solver_tolerance),
            nu_n=float(args.nu_n),
            skip_existing=bool(args.skip_existing),
            verbose=bool(args.verbose_sfincs),
            preferred_er=float(args.profile_root_preference),
            root_refine_width=float(args.convergence_root_refine_width),
            root_refine_max_iter=int(args.convergence_root_refine_max_iter),
            scan_root=out_dir
            / (
                "sfincs_radial_er_scan_convergence_"
                + _resolution_tag(
                    ntheta=int(args.convergence_ntheta),
                    nzeta=int(args.convergence_nzeta),
                    nxi=int(args.convergence_nxi),
                    nl=int(args.convergence_nl),
                    nx=int(args.convergence_nx),
                )
            ),
        )
        accuracy = convergence_summary(
            baseline=profile,
            refined=convergence_profile,
            max_abs_er_tolerance=float(args.convergence_max_abs_er),
            max_abs_bootstrap_tolerance=float(args.convergence_max_abs_bootstrap),
        )
        print(f"Refined-grid convergence scan dir: {convergence_scan_dir.resolve()}")
        print(
            "Convergence summary: "
            f"surfaces={accuracy.get('surfaces_checked')}, "
            f"max |Delta Er|={accuracy.get('max_abs_er')}, "
            f"max |Delta Jbs|={accuracy.get('max_abs_bootstrap')}, "
            f"passed={accuracy.get('passed')}"
        )
    representative_r_n, records, representative_scan_dir = _nearest_surface_records(scans_by_radius, float(args.r_n))
    roots = _ambipolar_roots(records)
    figure_png, figure_pdf = plot_summary(
        records=records,
        roots=roots,
        profile=profile,
        convergence_profile=convergence_profile,
        accuracy=accuracy,
        vmec_summary=vmec_summary,
        out_dir=out_dir,
        stem=str(args.stem),
        representative_r_n=representative_r_n,
    )
    summary = build_summary(
        vmec_summary=vmec_summary,
        records=records,
        roots=roots,
        profile=profile,
        convergence_profile=convergence_profile,
        accuracy=accuracy,
        figure_png=figure_png,
        figure_pdf=figure_pdf,
        scan_dir=scan_dir,
        representative_r_n=representative_r_n,
    )
    summary_path = out_dir / f"{args.stem}_summary.json"
    _write_json(summary_path, summary)

    root_text = ", ".join(f"{root:.6g}" for root in roots) if roots else "not bracketed"
    profile_text = ", ".join(
        f"r_N={p.r_n:.2f}: {p.selected_ambipolar_er:.6g}"
        for p in profile
        if p.selected_ambipolar_er is not None
    )
    print(f"  Representative scan dir: {representative_scan_dir.resolve()}")
    print(f"  Ambipolar Er roots at r_N={representative_r_n:.3g}: {root_text}")
    print(f"  Selected radial Er profile: {profile_text if profile_text else 'not bracketed'}")
    print(f"  Figure PNG: {figure_png}")
    print(f"  Figure PDF: {figure_pdf}")
    print(f"  Summary JSON: {summary_path.resolve()}")
    if not roots:
        print("  Note: the selected Er scan did not bracket an ambipolar root.")
    if bool(args.require_convergence) and not bool(accuracy.get("passed", False)):
        print("  Convergence requirement failed; outputs were still written for inspection.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
