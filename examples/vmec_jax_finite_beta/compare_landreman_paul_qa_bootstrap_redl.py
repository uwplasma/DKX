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

Run the bounded SFINCS-JAX comparison with radial coverage and numerical
error bars from one real-space refinement and one velocity-space refinement:

    python examples/vmec_jax_finite_beta/compare_landreman_paul_qa_bootstrap_redl.py \
      --run-sfincs --with-errorbars \
      --r-n-values 0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
      --n-lambda 16 \
      --ntheta 13 --nzeta 13 --nxi 13 --nl 13 --nx 13 \
      --real-ntheta 15 --real-nzeta 15 \
      --velocity-nxi 15 --velocity-nl 14 --velocity-nx 14 \
      --solver-tolerance 1e-6
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
from sfincs_jax.geometry.vmec_wout import read_vmec_wout  # noqa: E402


ELEMENTARY_CHARGE = 1.602176634e-19
ONE_KEV_J = 1.602176634e-16
PROTON_MASS_KG = 1.67262192369e-27
DEFAULT_ELECTRON_MHAT = 1.0 / 1836.15267343
DEFAULT_R_N_VALUES = "0.20,0.30,0.40,0.50,0.60,0.70,0.80"
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
    parser.add_argument(
        "--with-errorbars",
        action="store_true",
        help=(
            "Run/read two additional convergence probes and plot numerical error bars: "
            "one real-space refinement and one velocity-space refinement."
        ),
    )
    parser.add_argument("--real-ntheta", type=int, default=None, help="Real-space refinement Ntheta. Defaults to Ntheta+2.")
    parser.add_argument("--real-nzeta", type=int, default=None, help="Real-space refinement Nzeta. Defaults to Nzeta+2.")
    parser.add_argument("--velocity-nxi", type=int, default=None, help="Velocity-space refinement Nxi. Defaults to Nxi+2.")
    parser.add_argument("--velocity-nl", type=int, default=None, help="Velocity-space refinement NL. Defaults to NL+1.")
    parser.add_argument("--velocity-nx", type=int, default=None, help="Velocity-space refinement Nx. Defaults to Nx+1.")
    parser.add_argument("--solver-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--ne-coeffs", default=DEFAULT_NE_COEFFS, help="Redl electron-density polynomial in s, SI units.")
    parser.add_argument("--te-coeffs", default=DEFAULT_TE_COEFFS, help="Redl electron-temperature polynomial in s, eV.")
    parser.add_argument("--ti-coeffs", default=None, help="Ion-temperature polynomial in s, eV. Defaults to Te.")
    parser.add_argument("--zeff-coeffs", default="1.0", help="Zeff polynomial in s.")
    parser.add_argument("--helicity-n", type=int, default=0, help="Redl helicity_n. QA uses 0.")
    parser.add_argument("--n-lambda", type=int, default=32, help="Pitch integral quadrature for Redl trapped fraction.")
    parser.add_argument(
        "--collision-operator",
        type=int,
        choices=(0, 1),
        default=0,
        help="SFINCS-JAX collisionOperator: 0=full Fokker-Planck, 1=pitch-angle scattering.",
    )
    parser.add_argument("--ion-mhat", type=float, default=1.0, help="Ion mass normalized to SFINCS m_bar.")
    parser.add_argument(
        "--electron-mhat",
        type=float,
        default=DEFAULT_ELECTRON_MHAT,
        help="Electron mass normalized to SFINCS m_bar.",
    )
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
    ion_mhat: float,
    electron_mhat: float,
    dn_hat_dpsi_n: float,
    dt_i_hat_dpsi_n: float,
    dt_e_hat_dpsi_n: float,
    collision_operator: int,
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
  mHats = {float(ion_mhat):.16g} {float(electron_mhat):.16g}
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
  collisionOperator = {int(collision_operator)}
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
                ion_mhat=float(args.ion_mhat),
                electron_mhat=float(args.electron_mhat),
                dn_hat_dpsi_n=dn_hat,
                dt_i_hat_dpsi_n=dt_i_hat,
                dt_e_hat_dpsi_n=dt_e_hat,
                collision_operator=int(args.collision_operator),
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


def _scan_args_with_resolution(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    ntheta: int,
    nzeta: int,
    nxi: int,
    nl: int,
    nx: int,
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "out_dir": out_dir,
            "ntheta": int(ntheta),
            "nzeta": int(nzeta),
            "nxi": int(nxi),
            "nl": int(nl),
            "nx": int(nx),
        }
    )
    return argparse.Namespace(**values)


def _resolution_dict(args: argparse.Namespace) -> dict[str, int]:
    return {
        "Ntheta": int(args.ntheta),
        "Nzeta": int(args.nzeta),
        "Nxi": int(args.nxi),
        "NL": int(args.nl),
        "Nx": int(args.nx),
    }


def _sfincs_profile_from_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [np.nan if row.get("sfincs_j_parallel_si") is None else float(row["sfincs_j_parallel_si"]) for row in rows],
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
    redl: dict[str, Any],
    args: argparse.Namespace,
    wout_path: Path,
    scale: float,
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not bool(args.with_errorbars):
        return None

    real_ntheta = int(args.real_ntheta) if args.real_ntheta is not None else int(args.ntheta) + 2
    real_nzeta = int(args.real_nzeta) if args.real_nzeta is not None else int(args.nzeta) + 2
    velocity_nxi = int(args.velocity_nxi) if args.velocity_nxi is not None else int(args.nxi) + 2
    velocity_nl = int(args.velocity_nl) if args.velocity_nl is not None else int(args.nl) + 1
    velocity_nx = int(args.velocity_nx) if args.velocity_nx is not None else int(args.nx) + 1

    real_args = _scan_args_with_resolution(
        args,
        out_dir=args.out_dir / "sfincs_jax_points_real_space_refined",
        ntheta=real_ntheta,
        nzeta=real_nzeta,
        nxi=int(args.nxi),
        nl=int(args.nl),
        nx=int(args.nx),
    )
    velocity_args = _scan_args_with_resolution(
        args,
        out_dir=args.out_dir / "sfincs_jax_points_velocity_space_refined",
        ntheta=int(args.ntheta),
        nzeta=int(args.nzeta),
        nxi=velocity_nxi,
        nl=velocity_nl,
        nx=velocity_nx,
    )

    print(
        "Running/reading convergence probes: "
        f"real-space={_resolution_dict(real_args)}, velocity-space={_resolution_dict(velocity_args)}",
        flush=True,
    )
    real_rows = _run_or_read_sfincs(redl=redl, args=real_args, wout_path=wout_path, scale=scale)
    velocity_rows = _run_or_read_sfincs(redl=redl, args=velocity_args, wout_path=wout_path, scale=scale)

    baseline = _sfincs_profile_from_rows(baseline_rows)
    real = _sfincs_profile_from_rows(real_rows)
    velocity = _sfincs_profile_from_rows(velocity_rows)
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
        "sfincs_j_parallel_si_errorbar": _jsonify_array(error),
        "sfincs_j_parallel_si_errorbar_rel_to_baseline": _jsonify_array(rel_to_baseline),
        "real_space_delta_A_per_m2": _jsonify_array(real_delta),
        "velocity_space_delta_A_per_m2": _jsonify_array(velocity_delta),
        "real_space": real_rows,
        "velocity_space": velocity_rows,
    }


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
    convergence: dict[str, Any] | None = None,
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
            "This diagnostic compares a SFINCS-JAX RHSMode=1 kinetic solve to the "
            "Redl/Sauter analytic bootstrap-current fit. It is a normalization and trend "
            "diagnostic, not a Redl-parity claim: SFINCS-JAX uses the supplied nu_n, "
            "profile-gradient, and 3D drift-kinetic contract, while Redl uses local fitted "
            "nu_* coefficients for tokamak/quasisymmetric geometry."
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
            "collision_operator": int(args.collision_operator),
            "ion_mhat": float(args.ion_mhat),
            "electron_mhat": float(args.electron_mhat),
        },
        "normalization": {
            "sfincs_current_scale_A_per_m2": float(scale),
            "density_bar_m^-3": float(args.density_bar),
            "temperature_bar_keV": float(args.temperature_bar_kev),
            "mass_bar_kg": float(args.mass_bar_kg),
            "sfincs_si_formula": "FSABjHatOverRootFSAB2 * e * n_bar * sqrt(2*T_bar/m_bar)",
            "redl_geometry_B_convention": (
                "vmec_jax and SIMSOPT evaluate Redl trapped-particle geometry with the "
                "physical |B| = sqrt(2*(bsq - p)) from VMEC half-mesh bsq = |B|^2/2 + p."
            ),
        },
        "collisionality_contract": {
            "sfincs_nu_n": float(args.nu_n),
            "redl_nu_e_star": _jsonify_array(redl["nu_e_star"]),
            "redl_nu_i_star": _jsonify_array(redl["nu_i_star"]),
            "note": (
                "RHSMode=1 SFINCS-JAX uses the input nu_n together with dimensional "
                "species profiles and gradients; the Redl fit reports local fitted "
                "electron/ion collisionalities nu_* on each selected surface."
            ),
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
    if convergence is not None:
        error = np.asarray(convergence["sfincs_j_parallel_si_errorbar"], dtype=float)
        finite_error = error[np.isfinite(error)]
        rel_error = np.asarray(convergence["sfincs_j_parallel_si_errorbar_rel_to_baseline"], dtype=float)
        finite_rel_error = rel_error[np.isfinite(rel_error)]
        payload["convergence_errorbars"] = convergence
        payload["comparison"]["max_errorbar_A_per_m2"] = None if finite_error.size == 0 else float(np.max(finite_error))
        payload["comparison"]["max_errorbar_rel_to_sfincs"] = None if finite_rel_error.size == 0 else float(np.max(finite_rel_error))
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
    errorbars = np.full_like(sfincs, np.nan, dtype=float)
    if payload.get("convergence_errorbars") is not None:
        raw = np.asarray(payload["convergence_errorbars"].get("sfincs_j_parallel_si_errorbar", []), dtype=float)
        if raw.shape == sfincs.shape:
            errorbars = raw
    rel = np.abs(sfincs - redl) / np.maximum(np.abs(redl), 1.0e-300)
    rel_errorbars = errorbars / np.maximum(np.abs(redl), 1.0e-300)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    axes[0].plot(s, redl, "o-", color="#0f4c81", lw=2.4, label="Redl formula")
    if np.any(np.isfinite(sfincs)):
        if np.any(np.isfinite(errorbars)):
            axes[0].errorbar(
                s,
                sfincs,
                yerr=errorbars,
                fmt="s--",
                color="#c0392b",
                lw=2.2,
                capsize=3.0,
                label="sfincs_jax (baseline +/- refinement)",
            )
        else:
            axes[0].plot(s, sfincs, "s--", color="#c0392b", lw=2.2, label="sfincs_jax")
    else:
        axes[0].text(0.5, 0.5, "Run with --run-sfincs\nto add SFINCS-JAX points", ha="center", va="center", transform=axes[0].transAxes)
    axes[0].axhline(0.0, color="0.25", lw=0.9)
    axes[0].set_xlabel(r"$s=\psi_N$")
    axes[0].set_ylabel(r"$\langle J\cdot B\rangle/\sqrt{\langle B^2\rangle}$ [A m$^{-2}$]")
    axes[0].set_title("Bootstrap-current profile")
    axes[0].legend(loc="best")

    if np.any(np.isfinite(rel)):
        if np.any(np.isfinite(rel_errorbars)):
            axes[1].errorbar(s, rel, yerr=rel_errorbars, fmt="o-", color="#1b9e77", lw=2.2, capsize=3.0)
            positive = rel[np.isfinite(rel) & (rel > 0.0)]
            if positive.size:
                axes[1].set_ylim(bottom=max(float(np.min(positive)) * 0.5, 1.0e-6))
            axes[1].set_yscale("log")
        else:
            axes[1].semilogy(s, rel, "o-", color="#1b9e77", lw=2.2)
    else:
        axes[1].text(0.5, 0.5, "No SFINCS-JAX outputs loaded", ha="center", va="center", transform=axes[1].transAxes)
    axes[1].set_xlabel(r"$s=\psi_N$")
    axes[1].set_ylabel("relative difference vs Redl fit")
    axes[1].set_title("Kinetic current vs Redl fit")
    stats = payload["comparison"]
    max_err = stats.get("max_errorbar_rel_to_sfincs")
    max_rel = stats.get("max_rel_diff")
    max_abs = stats.get("max_abs_diff_A_per_m2")
    max_rel_text = "n/a" if max_rel is None else f"{float(max_rel):.3g}"
    max_abs_text = "n/a" if max_abs is None else f"{float(max_abs):.3g}"
    max_err_text = "n/a" if max_err is None else f"{float(max_err):.3g}"
    note = (
        f"compared points: {stats['n_compared']}\n"
        f"max abs diff: {max_abs_text} A m^-2\n"
        f"max rel diff: {max_rel_text}\n"
        f"max num. bar/SFINCS: {max_err_text}"
    )
    axes[1].text(0.03, 0.05, note, transform=axes[1].transAxes, ha="left", va="bottom", bbox={"facecolor": "white", "alpha": 0.82})
    fig.suptitle("Landreman-Paul QA bootstrap current: kinetic solve and Redl fit", fontsize=13.5)
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
    convergence = _run_convergence_errorbars(
        redl=redl,
        args=args,
        wout_path=wout_path,
        scale=scale,
        baseline_rows=sfincs_rows,
    )

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
        convergence=convergence,
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
