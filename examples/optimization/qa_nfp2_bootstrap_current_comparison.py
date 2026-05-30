#!/usr/bin/env python
"""Compare QA-only and bootstrap-current-aware QA optimization proxies.

This script is a fast documentation and teaching artifact. It optimizes a
small differentiable QA nfp=2 proxy twice:

1. quasisymmetry, rotational-transform, and aspect-ratio targets only;
2. the same targets plus a small-bootstrap-current penalty.

The resulting figure shows the 3D last-closed-flux-surface proxy, LCFS
cross-sections, and normalized bootstrap-current proxy profile. It is not a
kinetic SFINCS solve. Accepted candidates should still be promoted to completed
``sfincs_jax scan-er`` runs before making transport claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=160, help="JAX proxy-optimization steps per candidate.")
    parser.add_argument("--learning-rate", type=float, default=0.06, help="Normalized gradient step size.")
    parser.add_argument("--bootstrap-weight", type=float, default=1.0, help="Small-bootstrap-current penalty weight.")
    parser.add_argument("--target-iota", type=float, default=0.42, help="Target rotational-transform proxy.")
    parser.add_argument("--target-aspect", type=float, default=6.0, help="Target aspect ratio.")
    parser.add_argument("--out-dir", type=Path, default=Path.cwd(), help="Output directory.")
    parser.add_argument("--stem", default="qa_nfp2_bootstrap_current_comparison", help="Output file stem.")
    parser.add_argument("--no-plots", action="store_true", help="Write JSON only.")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable summary.")
    return parser


def _setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.8,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _initial_params() -> jnp.ndarray:
    """Return a deterministic QA-like starting point.

    Parameters are normalized shaping controls:
    ``kappa_drive``, ``triangularity_drive``, ``axis_shift``, and two
    non-axisymmetric ripple amplitudes.
    """

    return jnp.asarray([0.20, -0.15, 0.20, 0.08, -0.06], dtype=jnp.float64)


def _proxy_metrics(
    params: jnp.ndarray,
    *,
    target_iota: float,
    target_aspect: float,
) -> dict[str, jnp.ndarray]:
    """Return differentiable scalar metrics for the QA proxy."""

    kappa_drive, triangularity_drive, axis_shift, ripple_1, ripple_2 = params
    iota = 0.30 + 0.35 * kappa_drive + 0.12 * triangularity_drive - 0.03 * axis_shift
    aspect = target_aspect + 0.20 * axis_shift + 0.04 * kappa_drive**2
    qa_error = ripple_1**2 + ripple_2**2
    shape_norm = jnp.sum(params**2)

    # This proxy is constructed to represent a bootstrap-current-sensitive
    # direction that is not removed by QA/iota/aspect constraints alone.
    bootstrap_driver = (
        0.08
        + 1.10 * kappa_drive
        - 1.15 * triangularity_drive
        + 0.05 * axis_shift
        + 0.25 * ripple_1
        - 0.10 * ripple_2
    )
    return {
        "iota": iota,
        "aspect_ratio": aspect,
        "qa_error": qa_error,
        "shape_norm": shape_norm,
        "bootstrap_driver": bootstrap_driver,
        "iota_error": (iota - float(target_iota)) ** 2,
        "aspect_error": (aspect - float(target_aspect)) ** 2,
    }


def _objective(
    params: jnp.ndarray,
    *,
    bootstrap_weight: float,
    target_iota: float,
    target_aspect: float,
) -> jnp.ndarray:
    metrics = _proxy_metrics(params, target_iota=target_iota, target_aspect=target_aspect)
    return (
        200.0 * metrics["iota_error"]
        + 60.0 * metrics["aspect_error"]
        + 200.0 * metrics["qa_error"]
        + 0.025 * metrics["shape_norm"]
        + float(bootstrap_weight) * metrics["bootstrap_driver"] ** 2
    )


def _optimize(
    *,
    bootstrap_weight: float,
    steps: int,
    learning_rate: float,
    target_iota: float,
    target_aspect: float,
) -> dict[str, Any]:
    """Run a bounded normalized-gradient descent for the proxy objective."""

    params = _initial_params()
    value_and_grad = jax.jit(
        jax.value_and_grad(
            lambda q: _objective(
                q,
                bootstrap_weight=bootstrap_weight,
                target_iota=target_iota,
                target_aspect=target_aspect,
            )
        )
    )
    history: list[dict[str, float]] = []
    for step in range(int(steps) + 1):
        value, grad = value_and_grad(params)
        metrics = _proxy_metrics(params, target_iota=target_iota, target_aspect=target_aspect)
        history.append(
            {
                "step": float(step),
                "objective": float(value),
                "gradient_norm": float(jnp.linalg.norm(grad)),
                **{key: float(val) for key, val in metrics.items()},
            }
        )
        if step == int(steps):
            break
        params = params - float(learning_rate) * grad / (1.0 + jnp.linalg.norm(grad))

    return {
        "bootstrap_weight": float(bootstrap_weight),
        "initial_params": [float(x) for x in np.asarray(_initial_params())],
        "final_params": [float(x) for x in np.asarray(params)],
        "initial_metrics": history[0],
        "final_metrics": history[-1],
        "history": history,
    }


def _bootstrap_profile(params: np.ndarray, rho: np.ndarray) -> np.ndarray:
    metrics = _proxy_metrics(
        jnp.asarray(params, dtype=jnp.float64),
        target_iota=0.42,
        target_aspect=6.0,
    )
    driver = float(metrics["bootstrap_driver"])
    pressure_shape = (1.0 - 0.82 * rho**1.7) * (1.0 + 0.22 * rho) + 0.06
    return driver * pressure_shape


def _surface(
    params: np.ndarray,
    theta: np.ndarray,
    zeta: np.ndarray,
    *,
    rho: float = 1.0,
    target_aspect: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a simple VMEC-like QA nfp=2 LCFS proxy surface."""

    kappa_drive, triangularity_drive, axis_shift, ripple_1, ripple_2 = params
    theta2d, zeta2d = np.meshgrid(theta, zeta, indexing="ij")
    major_radius = float(target_aspect) + 0.20 * axis_shift
    elongation = 1.0 + 0.70 * kappa_drive
    triangularity = 0.24 * triangularity_drive
    helical = theta2d - 2.0 * zeta2d
    helical2 = 2.0 * theta2d - 2.0 * zeta2d

    minor_r = float(rho) * (
        1.0
        + triangularity * np.cos(theta2d)
        + 0.045 * ripple_1 * np.cos(helical)
        + 0.030 * ripple_2 * np.cos(helical2)
    )
    r_cyl = major_radius + minor_r * np.cos(theta2d)
    z_cyl = float(rho) * (
        elongation * np.sin(theta2d)
        + 0.045 * ripple_1 * np.sin(helical)
        + 0.030 * ripple_2 * np.sin(helical2)
    )
    x = r_cyl * np.cos(zeta2d)
    y = r_cyl * np.sin(zeta2d)

    b_proxy = (
        1.0
        + 0.050 * kappa_drive * np.cos(theta2d)
        + 0.035 * triangularity_drive * np.cos(2.0 * theta2d)
        + 0.045 * ripple_1 * np.cos(helical)
        + 0.030 * ripple_2 * np.cos(helical2)
    )
    return x, y, z_cyl, r_cyl, b_proxy


def _summary_stats(params: np.ndarray, *, target_iota: float, target_aspect: float) -> dict[str, Any]:
    metrics = _proxy_metrics(jnp.asarray(params), target_iota=target_iota, target_aspect=target_aspect)
    rho = np.linspace(0.05, 1.0, 96)
    current = _bootstrap_profile(params, rho)
    return {
        "params": [float(x) for x in params],
        "metrics": {key: float(value) for key, value in metrics.items()},
        "rho": [float(x) for x in rho],
        "bootstrap_current_normalized": [float(x) for x in current],
        "bootstrap_current_rms": float(np.sqrt(np.mean(current**2))),
        "bootstrap_current_max_abs": float(np.max(np.abs(current))),
    }


def _axis_scaled_3d(ax: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    xy_range = max(float(np.ptp(x)), float(np.ptp(y)))
    x_center = float(np.mean(x))
    y_center = float(np.mean(y))
    z_margin = 0.10 * max(float(np.ptp(z)), 1.0e-12)
    ax.set_xlim(x_center - 0.5 * xy_range, x_center + 0.5 * xy_range)
    ax.set_ylim(y_center - 0.5 * xy_range, y_center + 0.5 * xy_range)
    ax.set_zlim(float(np.min(z)) - z_margin, float(np.max(z)) + z_margin)
    ax.set_box_aspect((1.0, 1.0, 0.36))


def _write_figure(
    png_path: Path,
    pdf_path: Path,
    *,
    summary: dict[str, Any],
    target_aspect: float,
) -> None:
    _setup_mpl()
    qa_params = np.asarray(summary["qa_only"]["params"], dtype=float)
    boot_params = np.asarray(summary["qa_plus_bootstrap"]["params"], dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, 90, endpoint=True)
    zeta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=True)
    x0, y0, z0, r0, b0 = _surface(qa_params, theta, zeta, target_aspect=target_aspect)
    x1, y1, z1, r1, b1 = _surface(boot_params, theta, zeta, target_aspect=target_aspect)
    bmin = float(min(b0.min(), b1.min()))
    bmax = float(max(b0.max(), b1.max()))

    fig = plt.figure(figsize=(13.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    axes3d = [
        fig.add_subplot(gs[0, 0], projection="3d"),
        fig.add_subplot(gs[0, 1], projection="3d"),
    ]
    for ax, title, xyzb in [
        (axes3d[0], "A. QA/iota/aspect only", (x0, y0, z0, b0)),
        (axes3d[1], "B. QA + small bootstrap current", (x1, y1, z1, b1)),
    ]:
        x, y, z, b = xyzb
        colors = plt.cm.jet((b - bmin) / max(bmax - bmin, 1.0e-12))
        ax.plot_surface(x, y, z, facecolors=colors, rstride=2, cstride=2, linewidth=0.0, antialiased=False)
        ax.set_title(title)
        ax.set_xlabel("X/a")
        ax.set_ylabel("Y/a")
        ax.set_zlabel("Z/a")
        ax.view_init(elev=22, azim=-42)
        _axis_scaled_3d(ax, x, y, z)
    sm = mpl.cm.ScalarMappable(cmap="jet", norm=mpl.colors.Normalize(vmin=bmin, vmax=bmax))
    fig.colorbar(sm, ax=axes3d, shrink=0.70, pad=0.08, label=r"$B/B_{00}$ proxy")

    ax_lcfs = fig.add_subplot(gs[0, 2])
    zeta_cuts = [0.0, np.pi / 4.0, np.pi / 2.0]
    colors = ["#0f4c81", "#d95f02", "#1b9e77"]
    theta_cut = np.linspace(0.0, 2.0 * np.pi, 260, endpoint=True)
    for zeta_cut, color in zip(zeta_cuts, colors, strict=True):
        _, _, z_qa, r_qa, _ = _surface(qa_params, theta_cut, np.asarray([zeta_cut]), target_aspect=target_aspect)
        _, _, z_bo, r_bo, _ = _surface(boot_params, theta_cut, np.asarray([zeta_cut]), target_aspect=target_aspect)
        label = rf"$\zeta={zeta_cut / np.pi:.2g}\pi$"
        ax_lcfs.plot(r_qa[:, 0] - target_aspect, z_qa[:, 0], color=color, lw=1.6, label=label)
        ax_lcfs.plot(r_bo[:, 0] - target_aspect, z_bo[:, 0], color=color, lw=1.6, ls="--")
    ax_lcfs.set_aspect("equal", adjustable="box")
    ax_lcfs.set_title("C. LCFS cuts")
    ax_lcfs.set_xlabel(r"$R-R_0$ [$a$]")
    ax_lcfs.set_ylabel(r"$Z$ [$a$]")
    ax_lcfs.legend(title="solid: QA only\ndashed: +bootstrap", loc="best")

    ax_profile = fig.add_subplot(gs[1, 0])
    rho = np.asarray(summary["qa_only"]["rho"], dtype=float)
    j0 = np.asarray(summary["qa_only"]["bootstrap_current_normalized"], dtype=float)
    j1 = np.asarray(summary["qa_plus_bootstrap"]["bootstrap_current_normalized"], dtype=float)
    ax_profile.plot(rho, j0, color="#0f4c81", lw=2.6, label="QA/iota/aspect only")
    ax_profile.plot(rho, j1, color="#d95f02", lw=2.6, label="QA + bootstrap penalty")
    ax_profile.axhline(0.0, color="0.2", lw=1.0, alpha=0.65)
    ax_profile.set_title(r"D. Bootstrap-current proxy profile")
    ax_profile.set_xlabel(r"normalized toroidal-flux radius $\rho$")
    ax_profile.set_ylabel(r"$\langle\mathbf{J}\cdot\mathbf{B}\rangle/\sqrt{\langle B^2\rangle}$ [norm.]")
    ax_profile.legend(loc="best")

    ax_bar = fig.add_subplot(gs[1, 1])
    labels = ["RMS", "max abs"]
    qa_values = [
        summary["qa_only"]["bootstrap_current_rms"],
        summary["qa_only"]["bootstrap_current_max_abs"],
    ]
    boot_values = [
        summary["qa_plus_bootstrap"]["bootstrap_current_rms"],
        summary["qa_plus_bootstrap"]["bootstrap_current_max_abs"],
    ]
    x = np.arange(len(labels))
    width = 0.36
    ax_bar.bar(x - width / 2, qa_values, width=width, color="#0f4c81", label="QA only")
    ax_bar.bar(x + width / 2, boot_values, width=width, color="#d95f02", label="+bootstrap")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_title("E. Current reduction")
    ax_bar.set_ylabel("normalized magnitude")
    ax_bar.legend(loc="best")

    ax_metrics = fig.add_subplot(gs[1, 2])
    metric_labels = [r"$|\iota-\iota_0|$", r"$|A-6|$", "QA error", "current RMS"]
    qa_metrics = summary["qa_only"]["metrics"]
    boot_metrics = summary["qa_plus_bootstrap"]["metrics"]
    qa_metric_values = [
        abs(qa_metrics["iota"] - summary["targets"]["iota"]),
        abs(qa_metrics["aspect_ratio"] - summary["targets"]["aspect_ratio"]),
        qa_metrics["qa_error"],
        summary["qa_only"]["bootstrap_current_rms"],
    ]
    boot_metric_values = [
        abs(boot_metrics["iota"] - summary["targets"]["iota"]),
        abs(boot_metrics["aspect_ratio"] - summary["targets"]["aspect_ratio"]),
        boot_metrics["qa_error"],
        summary["qa_plus_bootstrap"]["bootstrap_current_rms"],
    ]
    y = np.arange(len(metric_labels))
    ax_metrics.barh(y + width / 2, qa_metric_values, height=width, color="#0f4c81", label="QA only")
    ax_metrics.barh(y - width / 2, boot_metric_values, height=width, color="#d95f02", label="+bootstrap")
    ax_metrics.set_yticks(y)
    ax_metrics.set_yticklabels(metric_labels)
    ax_metrics.set_xscale("log")
    ax_metrics.set_title("F. Constraint audit")
    ax_metrics.set_xlabel("proxy metric")
    ax_metrics.legend(loc="best")

    fig.suptitle(
        "QA nfp=2 optimization: adding a bootstrap-current objective changes the trade space",
        fontsize=13.5,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    jax.config.update("jax_enable_x64", True)

    qa_result = _optimize(
        bootstrap_weight=0.0,
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        target_iota=float(args.target_iota),
        target_aspect=float(args.target_aspect),
    )
    boot_result = _optimize(
        bootstrap_weight=float(args.bootstrap_weight),
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        target_iota=float(args.target_iota),
        target_aspect=float(args.target_aspect),
    )
    qa_summary = _summary_stats(
        np.asarray(qa_result["final_params"], dtype=float),
        target_iota=float(args.target_iota),
        target_aspect=float(args.target_aspect),
    )
    boot_summary = _summary_stats(
        np.asarray(boot_result["final_params"], dtype=float),
        target_iota=float(args.target_iota),
        target_aspect=float(args.target_aspect),
    )
    reduction = 1.0 - boot_summary["bootstrap_current_rms"] / max(qa_summary["bootstrap_current_rms"], 1.0e-300)
    summary = {
        "workflow": "sfincs_jax_qa_bootstrap_current_proxy_comparison",
        "claim_boundary": (
            "Fast differentiable QA optimization proxy. The plotted bootstrap current is a normalized "
            "optimizer-steering proxy, not a high-fidelity SFINCS kinetic current. Promote accepted "
            "candidates with completed sfincs_jax scan-er outputs before publication claims."
        ),
        "nfp": 2,
        "targets": {
            "iota": float(args.target_iota),
            "aspect_ratio": float(args.target_aspect),
        },
        "qa_only_optimizer": qa_result,
        "qa_plus_bootstrap_optimizer": boot_result,
        "qa_only": qa_summary,
        "qa_plus_bootstrap": boot_summary,
        "comparison": {
            "bootstrap_current_rms_reduction_fraction": float(reduction),
            "bootstrap_current_rms_ratio": float(
                boot_summary["bootstrap_current_rms"] / max(qa_summary["bootstrap_current_rms"], 1.0e-300)
            ),
            "qa_error_ratio": float(
                boot_summary["metrics"]["qa_error"] / max(qa_summary["metrics"]["qa_error"], 1.0e-300)
            ),
        },
        "promotion_plan": {
            "required_gates": [
                "write the accepted QA-plus-bootstrap candidate as a VMEC equilibrium",
                "run sfincs_jax scan-er at selected radii and electric-field values",
                "check residual convergence and CPU/GPU agreement",
                "compare FSABjHatOverRootFSAB2 and fluxes against SFINCS Fortran v3 when in shared scope",
                "run radial and velocity-space convergence before using the current profile in a paper",
            ],
        },
    }

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.stem}.json"
    png_path = out_dir / f"{args.stem}.png"
    pdf_path = out_dir / f"{args.stem}.pdf"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.no_plots:
        _write_figure(png_path, pdf_path, summary=summary, target_aspect=float(args.target_aspect))

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("QA nfp=2 bootstrap-current comparison proxy complete")
        print(f"  RMS current ratio: {summary['comparison']['bootstrap_current_rms_ratio']:.3e}")
        print(f"  RMS current reduction: {100.0 * reduction:.1f}%")
        print(f"  summary: {json_path}")
        if not args.no_plots:
            print(f"  figure:  {png_path}")
            print(f"  pdf:     {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
