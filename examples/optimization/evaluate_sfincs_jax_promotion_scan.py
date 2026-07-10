#!/usr/bin/env python
"""Evaluate high-fidelity sfincs_jax promotion gates for an optimization scan.

Use this script after a proxy optimization has selected a candidate geometry and
you have run an ``sfincs_jax scan-er`` over the candidate.  The script reads the
completed ``sfincsOutput.h5`` files, evaluates ambipolarity, bootstrap current,
main-species heat/particle flux, impurity-flux selectivity, and residual gates,
then writes a JSON audit and publication-style plot.

If ``--scan-dir`` is omitted, a tiny synthetic SFINCS-style scan is generated in
the output directory so the plotting and gate logic can be demonstrated quickly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.io import write_sfincs_h5  # noqa: E402
from sfincs_jax.workflows.optimization import evaluate_sfincs_scan_promotion  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, help="Completed sfincs_jax scan-er directory.")
    parser.add_argument("--out-dir", type=Path, default=Path.cwd(), help="Directory for JSON/PNG/PDF outputs.")
    parser.add_argument("--stem", default="qa_nfp2_sfincs_jax_promotion_scan", help="Output file stem.")
    parser.add_argument("--require-electron-root", action="store_true", default=True)
    parser.add_argument("--allow-no-electron-root", action="store_false", dest="require_electron_root")
    parser.add_argument(
        "--impurity-species-index",
        type=int,
        default=None,
        help=(
            "Optional species index for the flux-selectivity objective. "
            "Leave unset for two-species electron-root scans without an impurity."
        ),
    )
    parser.add_argument("--target-impurity-flux", type=float, default=0.01)
    parser.add_argument("--bootstrap-normalizer", type=float, default=1.0)
    parser.add_argument(
        "--allow-missing-residuals",
        action="store_false",
        dest="require_residuals",
        help=(
            "Do not fail when linear residual diagnostics are absent. Use this "
            "only for reference outputs, such as SFINCS Fortran v3 files, that "
            "do not write JAX residual datasets."
        ),
    )
    parser.set_defaults(require_residuals=True)
    parser.add_argument("--json", action="store_true", help="Print summary JSON.")
    return parser


def _setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10.3,
            "axes.labelsize": 10.3,
            "axes.titlesize": 11.2,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _write_demo_scan(scan_dir: Path) -> None:
    scan_dir.mkdir(parents=True, exist_ok=True)
    z_s = np.asarray([1.0, -1.0, 6.0], dtype=np.float64)
    for er, current in [(-3.0, -2.0), (-1.0, -0.8), (1.0, 0.35), (3.0, 1.9)]:
        run_dir = scan_dir / f"Er{er:.4g}"
        run_dir.mkdir(parents=True, exist_ok=True)
        gamma_i = 0.02 * er
        gamma_e = 0.10 + 0.01 * er
        gamma_z = (current - gamma_i + gamma_e) / 6.0
        particle = np.asarray([[gamma_i], [gamma_e], [gamma_z]], dtype=np.float64)
        heat = np.asarray([[0.030 + 0.006 * er**2], [0.020 + 0.004 * er**2], [0.004]], dtype=np.float64)
        data = {
            "Er": np.asarray(er),
            "Nspecies": np.asarray(3, dtype=np.int32),
            "Zs": z_s,
            "includePhi1": np.asarray(0, dtype=np.int32),
            "particleFlux_vm_rHat": particle,
            "heatFlux_vm_rHat": heat,
            "FSABjHat": np.asarray([0.04 + 0.015 * er], dtype=np.float64),
            "FSABjHatOverRootFSAB2": np.asarray([0.04 + 0.015 * er], dtype=np.float64),
            "linearSolverResidualNorm": np.asarray(1.0e-10, dtype=np.float64),
            "linearSolverResidualTarget": np.asarray(1.0e-8, dtype=np.float64),
        }
        write_sfincs_h5(path=run_dir / "sfincsOutput.h5", data=data, overwrite=True, fortran_layout=False)


def _plot_summary(path_png: Path, path_pdf: Path, payload: dict) -> None:
    _setup_mpl()
    runs = sorted(payload["runs"], key=lambda run: run["er"])
    er = np.asarray([run["er"] for run in runs], dtype=float)
    current = np.asarray([run["radial_current"] for run in runs], dtype=float)
    bootstrap = np.asarray([run["bootstrap_current"] for run in runs], dtype=float)
    particle = np.asarray([run["particle_flux"] for run in runs], dtype=float)
    heat = np.asarray([run["heat_flux"] for run in runs], dtype=float)
    selected = payload.get("selected_root")

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(er, current, marker="o", color="#0f4c81", lw=2.2)
    ax.axhline(0.0, color="0.25", lw=1.0)
    if selected is not None:
        ax.axvline(selected["er"], color="#d95f02", ls="--", label=f"{selected['root_type']} root")
        ax.legend(loc="best")
    ax.set_title("A. Ambipolarity")
    ax.set_xlabel(r"$E_r$")
    ax.set_ylabel(r"$\sum_s Z_s\Gamma_s$")

    ax = axes[0, 1]
    ax.plot(er, bootstrap, marker="s", color="#1b9e77", lw=2.2)
    if selected is not None:
        ax.axvline(selected["er"], color="#d95f02", ls="--")
    ax.set_title("B. Bootstrap current")
    ax.set_xlabel(r"$E_r$")
    ax.set_ylabel(r"$\langle J\cdot B\rangle/\sqrt{\langle B^2\rangle}$")

    ax = axes[1, 0]
    for idx in range(particle.shape[1]):
        ax.plot(er, particle[:, idx], marker="o", lw=2.0, label=f"species {idx}")
    if selected is not None:
        ax.axvline(selected["er"], color="#d95f02", ls="--")
    ax.set_title("C. Particle flux")
    ax.set_xlabel(r"$E_r$")
    ax.set_ylabel(r"$\Gamma_s$")
    ax.legend(loc="best")

    ax = axes[1, 1]
    for idx in range(heat.shape[1]):
        ax.plot(er, heat[:, idx], marker="^", lw=2.0, label=f"species {idx}")
    status = payload["gate_status"]
    ax.text(
        0.04,
        0.95,
        f"gate: {status}\nbootstrap J={payload['bootstrap_objective']:.2e}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85},
    )
    if selected is not None:
        ax.axvline(selected["er"], color="#d95f02", ls="--")
    ax.set_title("D. Heat flux and gate")
    ax.set_xlabel(r"$E_r$")
    ax.set_ylabel(r"$Q_s$")
    ax.legend(loc="best")

    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scan_dir = args.scan_dir.resolve() if args.scan_dir is not None else out_dir / f"{args.stem}_demo_scan"
    if args.scan_dir is None:
        _write_demo_scan(scan_dir)

    summary = evaluate_sfincs_scan_promotion(
        scan_dir,
        require_electron_root=bool(args.require_electron_root),
        impurity_species_index=(
            None if args.impurity_species_index is None else int(args.impurity_species_index)
        ),
        target_impurity_flux=float(args.target_impurity_flux),
        bootstrap_normalizer=float(args.bootstrap_normalizer),
        require_residuals=bool(args.require_residuals),
    )
    payload = summary.as_dict()
    json_path = out_dir / f"{args.stem}.json"
    png_path = out_dir / f"{args.stem}.png"
    pdf_path = out_dir / f"{args.stem}.pdf"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot_summary(png_path, pdf_path, payload)

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("sfincs_jax optimization promotion scan complete")
        print(f"  gate:    {payload['gate_status']}")
        print(f"  scan:    {scan_dir}")
        print(f"  summary: {json_path}")
        print(f"  figure:  {png_path}")
        print(f"  pdf:     {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
