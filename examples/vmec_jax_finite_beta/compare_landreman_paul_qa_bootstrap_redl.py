#!/usr/bin/env python
"""Compare SFINCS-JAX and Redl bootstrap current for Landreman-Paul QA.

This is a post-processing / validation example, not an optimization script.  It
loads a Landreman-Paul QA VMEC equilibrium, evaluates the Redl et al. bootstrap
current formula through ``vmec_jax``, optionally runs ``sfincs_jax`` at the same
surfaces, and plots the normalized bootstrap-current profiles side by side.

The plotted comparison uses

* Redl: ``<J.B> / sqrt(<B.B>)`` from ``vmec_jax.redl_bootstrap_jdotb``.
* SFINCS-JAX: ``FSABjHatOverRootFSAB2`` converted to SI using the standard
  ``e n_bar sqrt(2 T_bar / m_bar)`` scale.

Run a fast Redl-only plot from the ``sfincs_jax`` repository root:

    python examples/vmec_jax_finite_beta/compare_landreman_paul_qa_bootstrap_redl.py --skip-sfincs

Run the bounded SFINCS-JAX comparison too:

    python examples/vmec_jax_finite_beta/compare_landreman_paul_qa_bootstrap_redl.py --run-sfincs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5  # noqa: E402
from sfincs_jax.vmec_wout import read_vmec_wout  # noqa: E402


ELEMENTARY_CHARGE = 1.602176634e-19
ONE_KEV_J = 1.602176634e-16
PROTON_MASS_KG = 1.67262192369e-27
DEFAULT_R_N_VALUES = "0.30,0.50,0.70"
DEFAULT_NE_COEFFS = "3.0e20,0.0,0.0,0.0,0.0,-2.97e20"
DEFAULT_TE_COEFFS = "15.0e3,-14.85e3"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vmec-jax-root",
        type=Path,
        default=Path(os.environ.get("SFINCS_JAX_VMEC_JAX_ROOT", "/Users/rogeriojorge/local/vmec_jax")),
        help="Local vmec_jax checkout containing the Landreman-Paul QA example data.",
    )
    parser.add_argument(
        "--vmec-input",
        type=Path,
        default=None,
        help=(
            "VMEC input deck. Defaults to "
            "<vmec-jax-root>/examples/data/input.LandremanPaul2021_QA_reactorScale_lowres."
        ),
    )
    parser.add_argument(
        "--wout",
        type=Path,
        default=None,
        help=(
            "VMEC wout file. Defaults to "
            "<vmec-jax-root>/examples/data/wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/landreman_paul_qa_bootstrap_redl"))
    parser.add_argument("--stem", default="landreman_paul_qa_bootstrap_redl_comparison")
    parser.add_argument(
        "--r-n-values",
        default=DEFAULT_R_N_VALUES,
        help="Comma-separated SFINCS r_N surfaces. Redl is evaluated at psi_N=r_N^2.",
    )
    parser.add_argument("--er", type=float, default=0.0, help="SFINCS-JAX Er value used for each surface.")
    parser.add_argument("--nu-n", type=float, default=8.31565e-3, help="SFINCS-JAX normalized collisionality.")
    parser.add_argument("--run-sfincs", action="store_true", help="Run missing SFINCS-JAX points.")
    parser.add_argument("--skip-sfincs", action="store_true", help="Do not run SFINCS-JAX; plot Redl and existing outputs only.")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Reuse existing sfincsOutput.h5 files.")
    parser.add_argument("--force", action="store_true", help="Overwrite and rerun SFINCS-JAX outputs.")
    parser.add_argument("--ntheta", type=int, default=9)
    parser.add_argument("--nzeta", type=int, default=9)
    parser.add_argument("--nxi", type=int, default=9)
    parser.add_argument("--nl", type=int, default=4)
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--ne-coeffs", default=DEFAULT_NE_COEFFS, help="Redl electron-density polynomial in s, SI units.")
    parser.add_argument("--te-coeffs", default=DEFAULT_TE_COEFFS, help="Redl electron-temperature polynomial in s, eV.")
    parser.add_argument("--ti-coeffs", default=None, help="Ion-temperature polynomial in s, eV. Defaults to Te.")
    parser.add_argument("--zeff-coeffs", default="1.0", help="Zeff polynomial in s.")
    parser.add_argument("--helicity-n", type=int, default=0, help="Redl helicity_n. QA uses 0.")
    parser.add_argument("--n-lambda", type=int, default=32, help="Pitch integral quadrature for Redl trapped fraction.")
    parser.add_argument("--density-bar", type=float, default=1.0e20, help="SFINCS density normalization n_bar [m^-3].")
    parser.add_argument("--temperature-bar-kev", type=float, default=1.0, help="SFINCS temperature normalization [keV].")
    parser.add_argument("--mass-bar-kg", type=float, default=PROTON_MASS_KG, help="SFINCS reference mass [kg].")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the summary JSON.")
    return parser


def _parse_csv_floats(text: str | None, *, name: str) -> list[float] | None:
    if text is None:
        return None
    values = [float(piece.strip()) for piece in str(text).split(",") if piece.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one numeric value.")
    return values


def _parse_r_n_values(text: str) -> np.ndarray:
    values = np.asarray(_parse_csv_floats(text, name="--r-n-values"), dtype=float)
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("All --r-n-values entries must satisfy 0 < r_N < 1.")
    return np.asarray(sorted(set(float(v) for v in values)), dtype=float)


def _require_vmec_jax(root: Path):
    root = root.expanduser().resolve()
    if (root / "vmec_jax" / "__init__.py").exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        import vmec_jax as vj
        from vmec_jax.finite_beta import redl_bootstrap_geometry_from_state
        from vmec_jax.redl_bootstrap import polynomial_profile_and_derivative, redl_bootstrap_jdotb
        from vmec_jax.wout import state_from_wout
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit(
            "This example requires vmec_jax. Pass --vmec-jax-root or set SFINCS_JAX_VMEC_JAX_ROOT."
        ) from exc
    return {
        "vj": vj,
        "redl_bootstrap_geometry_from_state": redl_bootstrap_geometry_from_state,
        "redl_bootstrap_jdotb": redl_bootstrap_jdotb,
        "polynomial_profile_and_derivative": polynomial_profile_and_derivative,
        "state_from_wout": state_from_wout,
        "root": root,
    }


def _resolve_default_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = args.vmec_jax_root.expanduser().resolve()
    vmec_input = args.vmec_input or root / "examples" / "data" / "input.LandremanPaul2021_QA_reactorScale_lowres"
    wout = args.wout or root / "examples" / "data" / "wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"
    vmec_input = vmec_input.expanduser().resolve()
    wout = wout.expanduser().resolve()
    if not vmec_input.exists():
        raise FileNotFoundError(f"Missing VMEC input deck: {vmec_input}")
    if not wout.exists():
        raise FileNotFoundError(f"Missing VMEC wout file: {wout}")
    return vmec_input, wout


def _profile_and_derivative(poly_fn: Any, coeffs: list[float], s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, derivs = poly_fn(np.asarray(coeffs, dtype=float), np.asarray(s, dtype=float))
    return np.asarray(values, dtype=float), np.asarray(derivs, dtype=float)


def _sfincs_current_scale(*, density_bar: float, temperature_bar_kev: float, mass_bar_kg: float) -> float:
    return float(ELEMENTARY_CHARGE * density_bar * np.sqrt(2.0 * temperature_bar_kev * ONE_KEV_J / mass_bar_kg))


def _evaluate_redl(
    *,
    vmec: dict[str, Any],
    vmec_input: Path,
    wout_path: Path,
    s_values: np.ndarray,
    ne_coeffs: list[float],
    te_coeffs: list[float],
    ti_coeffs: list[float] | None,
    zeff_coeffs: list[float],
    helicity_n: int,
    n_lambda: int,
) -> dict[str, Any]:
    vj = vmec["vj"]
    cfg, indata = vj.load_config(str(vmec_input))
    static = vj.build_static(cfg)
    wout = vj.load_wout(wout_path)
    state = vmec["state_from_wout"](wout)
    geom = vmec["redl_bootstrap_geometry_from_state"](
        state=state,
        static=static,
        indata=indata,
        signgs=int(getattr(wout, "signgs", 1)),
        surfaces=tuple(float(s) for s in s_values),
        n_lambda=int(n_lambda),
    )
    jdotb_redl, details = vmec["redl_bootstrap_jdotb"](
        s=geom["s"],
        G=geom["G"],
        R=geom["R"],
        iota=geom["iota"],
        epsilon=geom["epsilon"],
        f_t=geom["f_t"],
        psi_edge=geom["psi_edge"],
        nfp=int(geom["nfp"]),
        helicity_n=int(helicity_n),
        ne_coeffs=ne_coeffs,
        Te_coeffs=te_coeffs,
        Ti_coeffs=ti_coeffs,
        Zeff_coeffs=zeff_coeffs,
    )
    fsa_b2 = np.asarray(geom["fsa_B2"], dtype=float)
    jdotb_redl = np.asarray(jdotb_redl, dtype=float)
    j_parallel = jdotb_redl / np.sqrt(np.maximum(fsa_b2, 1.0e-300))
    ne, dne_ds = _profile_and_derivative(vmec["polynomial_profile_and_derivative"], ne_coeffs, np.asarray(geom["s"]))
    te, dte_ds = _profile_and_derivative(vmec["polynomial_profile_and_derivative"], te_coeffs, np.asarray(geom["s"]))
    ti, dti_ds = _profile_and_derivative(
        vmec["polynomial_profile_and_derivative"],
        te_coeffs if ti_coeffs is None else ti_coeffs,
        np.asarray(geom["s"]),
    )
    return {
        "s": np.asarray(geom["s"], dtype=float),
        "r_n": np.sqrt(np.asarray(geom["s"], dtype=float)),
        "jdotb_redl": jdotb_redl,
        "j_parallel_redl_si": j_parallel,
        "fsa_B2": fsa_b2,
        "epsilon": np.asarray(geom["epsilon"], dtype=float),
        "f_t": np.asarray(geom["f_t"], dtype=float),
        "iota": np.asarray(geom["iota"], dtype=float),
        "G": np.asarray(geom["G"], dtype=float),
        "R": np.asarray(geom["R"], dtype=float),
        "ne": ne,
        "dne_ds": dne_ds,
        "te_eV": te,
        "dte_ds_eV": dte_ds,
        "ti_eV": ti,
        "dti_ds_eV": dti_ds,
        "nu_e_star": np.asarray(details["nu_e_star"], dtype=float),
        "nu_i_star": np.asarray(details["nu_i_star"], dtype=float),
        "L31": np.asarray(details["L31"], dtype=float),
        "L32": np.asarray(details["L32"], dtype=float),
        "L34": np.asarray(details["L34"], dtype=float),
        "alpha": np.asarray(details["alpha"], dtype=float),
        "nfp": int(geom["nfp"]),
    }


def _format_surface_dir(r_n: float) -> str:
    return f"rN{float(r_n):.4g}".replace(".", "p").replace("-", "m")


def _sfincs_template(
    *,
    wout_path: Path,
    r_n: float,
    er: float,
    nu_n: float,
    n_hat: float,
    t_i_hat: float,
    t_e_hat: float,
    dn_hat_dpsi_n: float,
    dt_i_hat_dpsi_n: float,
    dt_e_hat_dpsi_n: float,
    ntheta: int,
    nzeta: int,
    nxi: int,
    nl: int,
    nx: int,
    solver_tolerance: float,
) -> str:
    return f"""! Generated by examples/vmec_jax_finite_beta/compare_landreman_paul_qa_bootstrap_redl.py
&general
  RHSMode = 1
  saveMatricesAndVectorsInBinary = .false.
/

&geometryParameters
  geometryScheme = 5
  equilibriumFile = "{wout_path}"
  inputRadialCoordinate = 3
  inputRadialCoordinateForGradients = 1
  rN_wish = {float(r_n):.16g}
  VMECRadialOption = 0
/

&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.45509d-4
  nHats = {float(n_hat):.16g} {float(n_hat):.16g}
  THats = {float(t_i_hat):.16g} {float(t_e_hat):.16g}
  dNHatdpsiNs = {float(dn_hat_dpsi_n):.16g} {float(dn_hat_dpsi_n):.16g}
  dTHatdpsiNs = {float(dt_i_hat_dpsi_n):.16g} {float(dt_e_hat_dpsi_n):.16g}
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
"""


def _last_scalar(data: dict[str, Any], key: str) -> float:
    arr = np.asarray(data[key], dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return float(arr.reshape(-1)[-1])


def _run_or_read_sfincs(
    *,
    redl: dict[str, Any],
    args: argparse.Namespace,
    wout_path: Path,
    scale: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_root = args.out_dir / "sfincs_jax_points"
    if bool(args.run_sfincs) and not bool(args.skip_sfincs):
        aminor_p = float(read_vmec_wout(wout_path).aminor_p)
        if aminor_p <= 0.0:
            raise SystemExit(
                "SFINCS-JAX requires a VMEC wout with positive Aminor_p for "
                "radial-coordinate conversions. Use the default reactor-scale "
                "Landreman-Paul QA wout or pass --skip-sfincs for Redl-only plotting."
            )
    for idx, r_n in enumerate(np.asarray(redl["r_n"], dtype=float)):
        run_dir = run_root / _format_surface_dir(float(r_n))
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / "input.namelist"
        output_path = run_dir / "sfincsOutput.h5"
        n_hat = float(redl["ne"][idx] / float(args.density_bar))
        t_e_hat = float(redl["te_eV"][idx] / (1000.0 * float(args.temperature_bar_kev)))
        t_i_hat = float(redl["ti_eV"][idx] / (1000.0 * float(args.temperature_bar_kev)))
        dn_hat = float(redl["dne_ds"][idx] / float(args.density_bar))
        dt_e_hat = float(redl["dte_ds_eV"][idx] / (1000.0 * float(args.temperature_bar_kev)))
        dt_i_hat = float(redl["dti_ds_eV"][idx] / (1000.0 * float(args.temperature_bar_kev)))
        input_path.write_text(
            _sfincs_template(
                wout_path=wout_path,
                r_n=float(r_n),
                er=float(args.er),
                nu_n=float(args.nu_n),
                n_hat=n_hat,
                t_i_hat=t_i_hat,
                t_e_hat=t_e_hat,
                dn_hat_dpsi_n=dn_hat,
                dt_i_hat_dpsi_n=dt_i_hat,
                dt_e_hat_dpsi_n=dt_e_hat,
                ntheta=int(args.ntheta),
                nzeta=int(args.nzeta),
                nxi=int(args.nxi),
                nl=int(args.nl),
                nx=int(args.nx),
                solver_tolerance=float(args.solver_tolerance),
            ),
            encoding="utf-8",
        )
        should_run = bool(args.run_sfincs) and not bool(args.skip_sfincs)
        if output_path.exists() and not bool(args.force) and bool(args.skip_existing):
            should_run = False
        if should_run:
            print(f"Running SFINCS-JAX r_N={float(r_n):.4g} -> {output_path}", flush=True)
            t0 = time.perf_counter()
            write_sfincs_jax_output_h5(
                input_namelist=input_path,
                output_path=output_path,
                compute_solution=True,
                overwrite=True,
                verbose=True,
                differentiable=False,
            )
            elapsed_s = float(time.perf_counter() - t0)
        else:
            elapsed_s = 0.0
        if output_path.exists():
            data = read_sfincs_h5(output_path)
            fsab_over_root = _last_scalar(data, "FSABjHatOverRootFSAB2")
            fsab = _last_scalar(data, "FSABjHat")
            rows.append(
                {
                    "status": "loaded",
                    "r_n": float(r_n),
                    "s": float(redl["s"][idx]),
                    "input": str(input_path.resolve()),
                    "output": str(output_path.resolve()),
                    "elapsed_s": elapsed_s,
                    "FSABjHat": fsab,
                    "FSABjHatOverRootFSAB2": fsab_over_root,
                    "sfincs_j_parallel_si": float(fsab_over_root * scale),
                    "sfincs_jdotb_scaled": float(fsab * scale),
                }
            )
        else:
            rows.append(
                {
                    "status": "missing",
                    "r_n": float(r_n),
                    "s": float(redl["s"][idx]),
                    "input": str(input_path.resolve()),
                    "output": str(output_path.resolve()),
                    "elapsed_s": elapsed_s,
                    "FSABjHat": None,
                    "FSABjHatOverRootFSAB2": None,
                    "sfincs_j_parallel_si": None,
                    "sfincs_jdotb_scaled": None,
                }
            )
    return rows


def _jsonify_array(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]


def _write_summary(
    *,
    path: Path,
    args: argparse.Namespace,
    vmec_input: Path,
    wout_path: Path,
    redl: dict[str, Any],
    sfincs_rows: list[dict[str, Any]],
    scale: float,
) -> dict[str, Any]:
    sfincs_values = np.asarray(
        [np.nan if row["sfincs_j_parallel_si"] is None else float(row["sfincs_j_parallel_si"]) for row in sfincs_rows],
        dtype=float,
    )
    redl_values = np.asarray(redl["j_parallel_redl_si"], dtype=float)
    mask = np.isfinite(sfincs_values) & np.isfinite(redl_values)
    diff = sfincs_values[mask] - redl_values[mask]
    rel = np.abs(diff) / np.maximum(np.abs(redl_values[mask]), 1.0e-300) if np.any(mask) else np.asarray([])
    payload = {
        "workflow": "landreman_paul_qa_sfincs_jax_redl_bootstrap_current_comparison",
        "claim_boundary": (
            "This is a normalization and trend comparison between SFINCS-JAX and the Redl "
            "bootstrap-current fit for one profile contract; it is not an optimization script."
        ),
        "inputs": {
            "vmec_input": str(vmec_input),
            "wout": str(wout_path),
            "r_n_values": _jsonify_array(redl["r_n"]),
            "s_values": _jsonify_array(redl["s"]),
            "Er": float(args.er),
            "nu_n": float(args.nu_n),
            "ne_coeffs": _parse_csv_floats(args.ne_coeffs, name="--ne-coeffs"),
            "te_coeffs": _parse_csv_floats(args.te_coeffs, name="--te-coeffs"),
            "ti_coeffs": _parse_csv_floats(args.ti_coeffs, name="--ti-coeffs") if args.ti_coeffs else None,
            "zeff_coeffs": _parse_csv_floats(args.zeff_coeffs, name="--zeff-coeffs"),
        },
        "normalization": {
            "sfincs_current_scale_A_per_m2": float(scale),
            "density_bar_m^-3": float(args.density_bar),
            "temperature_bar_keV": float(args.temperature_bar_kev),
            "mass_bar_kg": float(args.mass_bar_kg),
            "sfincs_si_formula": "FSABjHatOverRootFSAB2 * e * n_bar * sqrt(2*T_bar/m_bar)",
        },
        "redl": {
            "jdotb_redl": _jsonify_array(redl["jdotb_redl"]),
            "j_parallel_redl_si": _jsonify_array(redl["j_parallel_redl_si"]),
            "fsa_B2": _jsonify_array(redl["fsa_B2"]),
            "epsilon": _jsonify_array(redl["epsilon"]),
            "f_t": _jsonify_array(redl["f_t"]),
            "iota": _jsonify_array(redl["iota"]),
            "nu_e_star": _jsonify_array(redl["nu_e_star"]),
            "nu_i_star": _jsonify_array(redl["nu_i_star"]),
            "L31": _jsonify_array(redl["L31"]),
            "L32": _jsonify_array(redl["L32"]),
            "L34": _jsonify_array(redl["L34"]),
        },
        "sfincs_jax": sfincs_rows,
        "comparison": {
            "n_compared": int(np.count_nonzero(mask)),
            "max_abs_diff_A_per_m2": None if not np.any(mask) else float(np.max(np.abs(diff))),
            "max_rel_diff": None if not np.any(mask) else float(np.max(rel)),
            "mean_rel_diff": None if not np.any(mask) else float(np.mean(rel)),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _setup_mpl() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.24,
        }
    )


def _write_plot(*, payload: dict[str, Any], png_path: Path, pdf_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_mpl()
    s = np.asarray(payload["inputs"]["s_values"], dtype=float)
    redl = np.asarray(payload["redl"]["j_parallel_redl_si"], dtype=float)
    sfincs = np.asarray(
        [
            np.nan if row["sfincs_j_parallel_si"] is None else float(row["sfincs_j_parallel_si"])
            for row in payload["sfincs_jax"]
        ],
        dtype=float,
    )
    rel = np.abs(sfincs - redl) / np.maximum(np.abs(redl), 1.0e-300)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    axes[0].plot(s, redl, "o-", color="#0f4c81", lw=2.4, label="Redl formula")
    if np.any(np.isfinite(sfincs)):
        axes[0].plot(s, sfincs, "s--", color="#c0392b", lw=2.2, label="sfincs_jax")
    else:
        axes[0].text(0.5, 0.5, "Run with --run-sfincs\nto add SFINCS-JAX points", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].axhline(0.0, color="0.25", lw=0.9)
    axes[0].set_xlabel(r"$s=\psi_N$")
    axes[0].set_ylabel(r"$\langle J\cdot B\rangle/\sqrt{\langle B^2\rangle}$ [A m$^{-2}$]")
    axes[0].set_title("Bootstrap-current profile")
    axes[0].legend(loc="best")

    if np.any(np.isfinite(rel)):
        axes[1].semilogy(s, rel, "o-", color="#1b9e77", lw=2.2)
    else:
        axes[1].text(0.5, 0.5, "No SFINCS-JAX outputs loaded", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_xlabel(r"$s=\psi_N$")
    axes[1].set_ylabel("relative difference")
    axes[1].set_title("SFINCS-JAX vs Redl")
    stats = payload["comparison"]
    note = (
        f"compared points: {stats['n_compared']}\n"
        f"max rel diff: {stats['max_rel_diff'] if stats['max_rel_diff'] is not None else 'n/a'}"
    )
    axes[1].text(0.03, 0.05, note, transform=axes[1].transAxes, ha="left", va="bottom", bbox={"facecolor": "white", "alpha": 0.82})
    fig.suptitle("Landreman-Paul QA bootstrap current: sfincs_jax vs Redl", fontsize=13.5)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)


def main() -> int:
    args = _build_parser().parse_args()
    if bool(args.skip_sfincs):
        args.run_sfincs = False
    if bool(args.force):
        args.skip_existing = False
    vmec_input, wout_path = _resolve_default_paths(args)
    vmec = _require_vmec_jax(args.vmec_jax_root)
    r_n = _parse_r_n_values(args.r_n_values)
    s_values = r_n * r_n
    ne_coeffs = _parse_csv_floats(args.ne_coeffs, name="--ne-coeffs")
    te_coeffs = _parse_csv_floats(args.te_coeffs, name="--te-coeffs")
    ti_coeffs = _parse_csv_floats(args.ti_coeffs, name="--ti-coeffs") if args.ti_coeffs else None
    zeff_coeffs = _parse_csv_floats(args.zeff_coeffs, name="--zeff-coeffs")

    print("Evaluating Redl bootstrap-current formula", flush=True)
    redl = _evaluate_redl(
        vmec=vmec,
        vmec_input=vmec_input,
        wout_path=wout_path,
        s_values=s_values,
        ne_coeffs=ne_coeffs,
        te_coeffs=te_coeffs,
        ti_coeffs=ti_coeffs,
        zeff_coeffs=zeff_coeffs,
        helicity_n=int(args.helicity_n),
        n_lambda=int(args.n_lambda),
    )
    scale = _sfincs_current_scale(
        density_bar=float(args.density_bar),
        temperature_bar_kev=float(args.temperature_bar_kev),
        mass_bar_kg=float(args.mass_bar_kg),
    )
    sfincs_rows = _run_or_read_sfincs(redl=redl, args=args, wout_path=wout_path, scale=scale)

    json_path = args.out_dir / f"{args.stem}.json"
    png_path = args.out_dir / f"{args.stem}.png"
    pdf_path = args.out_dir / f"{args.stem}.pdf"
    payload = _write_summary(
        path=json_path,
        args=args,
        vmec_input=vmec_input,
        wout_path=wout_path,
        redl=redl,
        sfincs_rows=sfincs_rows,
        scale=scale,
    )
    if not args.no_plots:
        _write_plot(payload=payload, png_path=png_path, pdf_path=pdf_path)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        compared = payload["comparison"]["n_compared"]
        print(f"Wrote {json_path}")
        if not args.no_plots:
            print(f"Wrote {png_path}")
            print(f"Wrote {pdf_path}")
        print(f"Compared SFINCS-JAX and Redl at {compared} surface(s).")
        if compared == 0:
            print("Run again with --run-sfincs to generate missing SFINCS-JAX points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
