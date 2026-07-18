#!/usr/bin/env python
"""Plot a VMEC-backed QA optimization and bootstrap-current diagnostic.

This script intentionally uses the real ``vmex`` QA optimization artifacts
from ``examples/optimization/QA_optimization.py``.  It does not construct a
surrogate stellarator or a surrogate rotational-transform model.  The checked
README/docs artifact is therefore tied to the VMEC equilibrium that targets
aspect ratio 5 and mean iota 0.41 in the upstream ``vmex`` script.

The current panel is a VMEC equilibrium diagnostic,
``jdotb / sqrt(bdotb)``.  It is useful for teaching how current-sensitive
optimization hooks are inspected, but it is not a completed kinetic
``dkx`` bootstrap-current claim.  Accepted candidates still need
completed radial/velocity convergence and CPU/GPU/Fortran kinetic gates before
being used as transport evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


_SCRIPT_NAME = "examples/optimization/QA_optimization.py"
_DEFAULT_STEM = "qa_nfp2_bootstrap_current_comparison"
_TARGET_ASPECT_FROM_VMEX = 5.0
_TARGET_IOTA_FROM_VMEX = 0.41


@dataclass(frozen=True)
class VmexContext:
    """Resolved local ``vmex`` module and repository paths."""

    module: Any
    root: Path | None
    qa_result_dir: Path


@dataclass(frozen=True)
class Candidate:
    """VMEC candidate loaded from a ``wout`` file."""

    label: str
    result_dir: Path
    wout_path: Path
    history_path: Path | None
    wout: Any
    history: dict[str, Any]
    metrics: dict[str, Any]
    profiles: dict[str, list[float]]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vmex-root",
        type=Path,
        default=None,
        help="Path to a local vmex checkout. Defaults to DKX_VMEX_ROOT, then common local paths.",
    )
    parser.add_argument(
        "--qa-result-dir",
        type=Path,
        default=None,
        help="Directory containing QA_optimization.py artifacts: history.json and wout_final.nc.",
    )
    parser.add_argument(
        "--comparison-result-dir",
        type=Path,
        default=None,
        help="Optional second vmex result directory, for example a QA run with JDotB or RedlBootstrapMismatch.",
    )
    parser.add_argument(
        "--comparison-label",
        default="QA + current objective",
        help="Label used when --comparison-result-dir is supplied.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path.cwd(), help="Output directory.")
    parser.add_argument("--stem", default=_DEFAULT_STEM, help="Output file stem.")
    parser.add_argument("--theta-points", type=int, default=72, help="Poloidal points for plotting.")
    parser.add_argument("--zeta-points", type=int, default=96, help="Toroidal points for plotting.")
    parser.add_argument("--no-plots", action="store_true", help="Write JSON only.")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable summary.")

    # Backward-compatible no-op flags retained so old README/demo commands fail
    # only when the real vmex artifacts are missing, not at argparse time.
    parser.add_argument("--steps", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-iota", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target-aspect", type=float, default=None, help=argparse.SUPPRESS)
    return parser


def _setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 11.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _candidate_vmex_roots(explicit: Path | None) -> list[Path | None]:
    roots: list[Path | None] = []
    if explicit is not None:
        roots.append(explicit.expanduser().resolve())
    env_root = os.environ.get("DKX_VMEX_ROOT")
    if env_root:
        roots.append(Path(env_root).expanduser().resolve())
    roots.extend(
        [
            Path("/Users/rogeriojorge/local/vmex"),
            Path("/Users/rogeriojorge/vmex"),
            None,
        ]
    )
    deduped: list[Path | None] = []
    seen: set[str] = set()
    for root in roots:
        key = "<import>" if root is None else str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _import_vmex(explicit_root: Path | None):
    errors: list[str] = []
    for root in _candidate_vmex_roots(explicit_root):
        try:
            if root is not None:
                if not root.exists():
                    errors.append(f"{root}: path does not exist")
                    continue
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
            import vmex as vj  # type: ignore[import-not-found]

            return vj, root
        except Exception as exc:  # pragma: no cover - diagnostic path
            errors.append(f"{root or 'import path'}: {type(exc).__name__}: {exc}")
    joined = "\n  ".join(errors)
    raise RuntimeError(
        "Could not import vmex. Install it or pass --vmex-root.\n"
        "Tried:\n  "
        f"{joined}"
    )


def _resolve_context(args: argparse.Namespace) -> VmexContext:
    vj, root = _import_vmex(args.vmex_root)
    if args.qa_result_dir is not None:
        qa_result_dir = args.qa_result_dir.expanduser().resolve()
    elif root is not None:
        qa_result_dir = root / "examples" / "optimization" / "results" / "qa_opt" / "ess"
    else:
        raise RuntimeError("Pass --qa-result-dir when vmex is imported from site-packages.")

    wout_path = qa_result_dir / "wout_final.nc"
    if not wout_path.exists():
        script_hint = (root / _SCRIPT_NAME) if root is not None else Path(_SCRIPT_NAME)
        raise FileNotFoundError(
            f"Missing {wout_path}. Generate it with the real vmex QA optimization first:\n"
            f"  cd {root or '<vmex checkout>'} && python {script_hint}"
        )
    return VmexContext(module=vj, root=root, qa_result_dir=qa_result_dir)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rho_from_wout(wout: Any) -> np.ndarray:
    phi = np.asarray(getattr(wout, "phi", []), dtype=float)
    if phi.size > 1 and np.isfinite(phi[-1]) and abs(float(phi[-1])) > 0.0:
        rho = np.sqrt(np.clip(phi / float(phi[-1]), 0.0, None))
        if np.all(np.isfinite(rho)):
            return rho
    iota = np.asarray(getattr(wout, "iotaf", []), dtype=float)
    n = int(iota.size) if iota.size else int(np.asarray(getattr(wout, "jdotb", [0.0])).size)
    return np.linspace(0.0, 1.0, max(n, 1))


def _current_profile(wout: Any) -> np.ndarray:
    jdotb = np.asarray(getattr(wout, "jdotb", []), dtype=float)
    bdotb = np.asarray(getattr(wout, "bdotb", np.ones_like(jdotb)), dtype=float)
    if jdotb.size == 0:
        return np.zeros((0,), dtype=float)
    root_b2 = np.sqrt(np.maximum(np.abs(bdotb), 1.0e-300))
    return np.divide(jdotb, root_b2, out=np.zeros_like(jdotb), where=root_b2 > 0.0)


def _candidate_metrics(wout: Any, history: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[float]]]:
    rho = _rho_from_wout(wout)
    iota = np.asarray(getattr(wout, "iotaf", getattr(wout, "iotas", np.zeros_like(rho))), dtype=float)
    current = _current_profile(wout)
    mask = np.isfinite(current) & (rho > 0.0)
    iota_mask = np.isfinite(iota) & (rho > 0.0)

    mean_iota = float(np.mean(iota[iota_mask])) if np.any(iota_mask) else float("nan")
    aspect = float(getattr(wout, "aspect", history.get("aspect_final", float("nan"))))
    current_rms = float(np.sqrt(np.mean(current[mask] ** 2))) if np.any(mask) else float("nan")
    current_max = float(np.max(np.abs(current[mask]))) if np.any(mask) else float("nan")
    history_iota = history.get("iota_final")
    history_aspect = history.get("aspect_final")

    metrics = {
        "aspect_ratio": aspect,
        "aspect_from_history": None if history_aspect is None else float(history_aspect),
        "mean_iota": mean_iota,
        "iota_from_history": None if history_iota is None else float(history_iota),
        "target_aspect_ratio": float(history.get("target_aspect", _TARGET_ASPECT_FROM_VMEX)),
        "target_iota": float(history.get("target_iota", _TARGET_IOTA_FROM_VMEX)),
        "jdotb_over_root_bdotb_rms": current_rms,
        "jdotb_over_root_bdotb_max_abs": current_max,
        "jdotb_rms": float(np.sqrt(np.mean(np.asarray(getattr(wout, "jdotb", [0.0]), dtype=float)[mask] ** 2)))
        if np.any(mask)
        else float("nan"),
        "n_radial": int(rho.size),
    }
    profiles = {
        "rho": [float(x) for x in rho],
        "iota": [float(x) for x in iota],
        "jdotb_over_root_bdotb": [float(x) for x in current],
    }
    return metrics, profiles


def _load_candidate(vj: Any, result_dir: Path, *, label: str) -> Candidate:
    result_dir = result_dir.expanduser().resolve()
    wout_path = result_dir / "wout_final.nc"
    if not wout_path.exists():
        raise FileNotFoundError(f"Missing VMEC wout file: {wout_path}")
    history_path = result_dir / "history.json"
    history = _load_json(history_path)
    wout = vj.load_wout(wout_path)
    metrics, profiles = _candidate_metrics(wout, history)
    return Candidate(
        label=label,
        result_dir=result_dir,
        wout_path=wout_path,
        history_path=history_path if history_path.exists() else None,
        wout=wout,
        history=history,
        metrics=metrics,
        profiles=profiles,
    )


def _gate(candidate: Candidate) -> dict[str, Any]:
    metrics = candidate.metrics
    aspect_err = abs(float(metrics["aspect_ratio"]) - float(metrics["target_aspect_ratio"]))
    iota_err = abs(float(metrics["mean_iota"]) - float(metrics["target_iota"]))
    return {
        "aspect_error": aspect_err,
        "iota_error": iota_err,
        "aspect_status": "pass" if aspect_err < 5.0e-3 else "fail",
        "iota_status": "pass" if iota_err < 2.0e-2 else "fail",
        "status": "pass" if aspect_err < 5.0e-3 and iota_err < 2.0e-2 else "fail",
        "note": "Gate uses the VMEC wout profile mean, not the simple-seed initial state.",
    }


def _comparison_metrics(primary: Candidate, comparison: Candidate | None) -> dict[str, Any]:
    if comparison is None:
        return {
            "status": "baseline_only",
            "reason": "No --comparison-result-dir was supplied.",
            "current_reduction_fraction": None,
        }
    base = float(primary.metrics["jdotb_over_root_bdotb_rms"])
    comp = float(comparison.metrics["jdotb_over_root_bdotb_rms"])
    ratio = comp / base if base > 0.0 else float("nan")
    return {
        "status": "comparison_loaded",
        "jdotb_over_root_bdotb_rms_ratio": ratio,
        "current_reduction_fraction": 1.0 - ratio,
        "candidate_has_smaller_current_rms": bool(comp < base),
    }


def _axis_scaled_3d(ax: Any, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    xy_range = max(float(np.ptp(x)), float(np.ptp(y)), 1.0e-12)
    x_center = 0.5 * (float(np.max(x)) + float(np.min(x)))
    y_center = 0.5 * (float(np.max(y)) + float(np.min(y)))
    z_margin = 0.08 * max(float(np.ptp(z)), 1.0e-12)
    ax.set_xlim(x_center - 0.5 * xy_range, x_center + 0.5 * xy_range)
    ax.set_ylim(y_center - 0.5 * xy_range, y_center + 0.5 * xy_range)
    ax.set_zlim(float(np.min(z)) - z_margin, float(np.max(z)) + z_margin)
    ax.set_box_aspect((1.0, 1.0, 0.34))


def _plot_surface(ax: Any, vj: Any, candidate: Candidate, *, theta_points: int, zeta_points: int) -> None:
    _, phi, r, z, b = vj.vmecplot2_lcfs_3d_grid(
        candidate.wout,
        s_index=-1,
        ntheta=int(theta_points),
        nzeta=int(zeta_points),
    )
    phi2d = np.broadcast_to(np.asarray(phi, dtype=float)[None, :], np.asarray(r).shape)
    x = np.asarray(r, dtype=float) * np.cos(phi2d)
    y = np.asarray(r, dtype=float) * np.sin(phi2d)
    z = np.asarray(z, dtype=float)
    b = np.asarray(b, dtype=float)
    norm = mpl.colors.Normalize(vmin=float(np.nanmin(b)), vmax=float(np.nanmax(b)))
    colors = plt.get_cmap("viridis")(norm(b))
    ax.plot_surface(x, y, z, facecolors=colors, linewidth=0.0, antialiased=False, shade=False, alpha=0.96)
    ax.view_init(elev=22, azim=34)
    ax.set_title("A. VMEC QA LCFS")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    _axis_scaled_3d(ax, x, y, z)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.62, pad=0.03)
    cbar.set_label(r"$|B|$")


def _plot_b_contour(ax: Any, vj: Any, candidate: Candidate, *, theta_points: int, zeta_points: int) -> None:
    theta, zeta, b = vj.vmecplot2_bmag_grid(
        candidate.wout,
        s_index=-1,
        ntheta=int(theta_points),
        nzeta=int(zeta_points),
        zeta_max=2.0 * np.pi / float(candidate.wout.nfp),
    )
    contour = ax.contourf(zeta, theta, np.asarray(b, dtype=float), levels=24, cmap="viridis")
    ax.set_title("B. LCFS field strength")
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$\theta$")
    cbar = plt.colorbar(contour, ax=ax, pad=0.02)
    cbar.set_label(r"$|B|$")


def _plot_lcfs_cuts(ax: Any, vj: Any, candidates: list[Candidate], *, theta_points: int) -> None:
    colors = ["#0f4c81", "#c0392b"]
    linestyles = ["-", "--"]
    for idx, cand in enumerate(candidates):
        for zeta_cut, alpha in [(0.0, 1.0), (np.pi / float(cand.wout.nfp), 0.5)]:
            _, _, r, z, _ = vj.vmecplot2_lcfs_3d_grid(
                cand.wout,
                s_index=-1,
                ntheta=int(theta_points),
                nzeta=2,
            )
            # The helper returns a fixed zeta grid. For deterministic cuts,
            # rebuild the nearest available column instead of interpolating.
            col = 0 if zeta_cut == 0.0 else -1
            label = cand.label if zeta_cut == 0.0 else None
            ax.plot(
                np.asarray(r)[:, col],
                np.asarray(z)[:, col],
                color=colors[idx % len(colors)],
                lw=1.8,
                ls=linestyles[idx % len(linestyles)],
                alpha=alpha,
                label=label,
            )
    ax.set_title("C. LCFS cuts")
    ax.set_xlabel("R")
    ax.set_ylabel("Z")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")


def _plot_profiles(ax_iota: Any, ax_current: Any, candidates: list[Candidate]) -> None:
    colors = ["#0f4c81", "#c0392b"]
    linestyles = ["-", "--"]
    reference_current = np.asarray(candidates[0].profiles["jdotb_over_root_bdotb"], dtype=float)
    reference_scale = max(float(np.nanmax(np.abs(reference_current))), 1.0e-300)
    for idx, cand in enumerate(candidates):
        rho = np.asarray(cand.profiles["rho"], dtype=float)
        iota = np.asarray(cand.profiles["iota"], dtype=float)
        current = np.asarray(cand.profiles["jdotb_over_root_bdotb"], dtype=float)
        color = colors[idx % len(colors)]
        ls = linestyles[idx % len(linestyles)]
        ax_iota.plot(rho, iota, color=color, lw=2.2, ls=ls, label=cand.label)
        ax_current.plot(rho, current / reference_scale, color=color, lw=2.2, ls=ls, label=cand.label)
    target_iota = float(candidates[0].metrics["target_iota"])
    ax_iota.axhline(target_iota, color="#333333", lw=1.0, ls=":", label=rf"target $\iota={target_iota:.2f}$")
    ax_iota.set_title(r"D. Rotational-transform profile")
    ax_iota.set_xlabel(r"$\rho=\sqrt{\Phi/\Phi_\mathrm{edge}}$")
    ax_iota.set_ylabel(r"$\iota$")
    ax_iota.legend(loc="best")
    ax_current.axhline(0.0, color="#333333", lw=0.8)
    ax_current.set_title(r"E. VMEC current diagnostic")
    ax_current.set_xlabel(r"$\rho=\sqrt{\Phi/\Phi_\mathrm{edge}}$")
    ax_current.set_ylabel(r"$J\cdot B/\sqrt{B\cdot B}$, normalized")
    ax_current.legend(loc="best")


def _plot_history(ax: Any, candidate: Candidate) -> None:
    history = candidate.history
    objective_history = history.get("objective_history") or history.get("objective_samples")
    if isinstance(objective_history, list) and objective_history:
        values = np.asarray(objective_history, dtype=float)
        ax.semilogy(np.arange(values.size), np.maximum(values, 1.0e-300), color="#253494", lw=2.2)
        ax.set_xlabel("objective sample")
        ax.set_ylabel("least-squares objective")
    else:
        values = np.asarray(
            [
                history.get("objective_initial", np.nan),
                history.get("objective_final", np.nan),
            ],
            dtype=float,
        )
        if np.all(np.isfinite(values)):
            ax.semilogy([0, 1], np.maximum(values, 1.0e-300), marker="o", color="#253494", lw=2.2)
            ax.set_xticks([0, 1], ["initial", "final"])
            ax.set_ylabel("least-squares objective")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, "No history.json found", ha="center", va="center")
            return
    gate = _gate(candidate)
    ax.set_title("F. QA_optimization.py audit")
    ax.text(
        0.02,
        0.04,
        "\n".join(
            [
                rf"$A={candidate.metrics['aspect_ratio']:.5f}$",
                rf"$\langle\iota\rangle={candidate.metrics['mean_iota']:.5f}$",
                f"gate: {gate['status']}",
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82, "edgecolor": "#bdbdbd"},
    )


def _write_figure(
    png_path: Path,
    pdf_path: Path,
    *,
    vj: Any,
    primary: Candidate,
    comparison: Candidate | None,
    theta_points: int,
    zeta_points: int,
) -> None:
    _setup_mpl()
    candidates = [primary] + ([comparison] if comparison is not None else [])
    fig = plt.figure(figsize=(14.2, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_b = fig.add_subplot(gs[0, 1])
    ax_lcfs = fig.add_subplot(gs[0, 2])
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_iota = fig.add_subplot(gs[1, 1])
    ax_current = fig.add_subplot(gs[1, 2])

    _plot_surface(ax3d, vj, primary, theta_points=theta_points, zeta_points=zeta_points)
    _plot_b_contour(ax_b, vj, primary, theta_points=theta_points, zeta_points=zeta_points)
    _plot_lcfs_cuts(ax_lcfs, vj, candidates, theta_points=theta_points)
    _plot_profiles(ax_iota, ax_current, candidates)
    _plot_history(ax_hist, primary)
    fig.suptitle(
        "VMEC-backed QA optimization diagnostic for dkx promotion",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _summary(
    *,
    ctx: VmexContext,
    primary: Candidate,
    comparison: Candidate | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    vmec_script = None if ctx.root is None else str((ctx.root / _SCRIPT_NAME).resolve())
    return {
        "workflow": "dkx_vmex_qa_optimization_current_diagnostic",
        "nfp": int(getattr(primary.wout, "nfp", 2)),
        "source": {
            "vmex_root": None if ctx.root is None else str(ctx.root),
            "qa_optimization_script": vmec_script,
            "qa_result_dir": str(primary.result_dir),
            "qa_wout": str(primary.wout_path),
            "qa_history": None if primary.history_path is None else str(primary.history_path),
            "comparison_result_dir": None if comparison is None else str(comparison.result_dir),
        },
        "targets": {
            "aspect_ratio": float(primary.metrics["target_aspect_ratio"]),
            "iota": float(primary.metrics["target_iota"]),
        },
        "qa_optimization": {
            "label": primary.label,
            "metrics": primary.metrics,
            "profiles": primary.profiles,
            "gate": _gate(primary),
        },
        "comparison_candidate": None
        if comparison is None
        else {
            "label": comparison.label,
            "metrics": comparison.metrics,
            "profiles": comparison.profiles,
            "gate": _gate(comparison),
        },
        "comparison": _comparison_metrics(primary, comparison),
        "claim_boundary": (
            "This is a VMEC equilibrium diagnostic generated from vmex "
            "QA_optimization.py outputs. The current profile is VMEC "
            "jdotb/sqrt(bdotb), not a completed high-fidelity dkx "
            "kinetic bootstrap-current claim."
        ),
        "promotion_plan": {
            "required_gates": [
                "preserve finite target iota and target aspect in the VMEC wout, not just in optimizer history",
                "if a current-aware VMEC candidate is supplied, require smaller VMEC jdotb/sqrt(bdotb) RMS without failing iota/aspect gates",
                "run completed dkx scan-er calculations on accepted equilibria",
                "check radial and velocity-space convergence of FSABjHatOverRootFSAB2",
                "check CPU/GPU agreement and SFINCS Fortran v3 agreement when the model scopes overlap",
            ],
            "example_commands": [
                "cd /path/to/vmex && python examples/optimization/QA_optimization.py",
                "python examples/optimization/qa_nfp2_bootstrap_current_comparison.py --vmex-root /path/to/vmex",
            ],
        },
        "plotting": {
            "theta_points": int(args.theta_points),
            "zeta_points": int(args.zeta_points),
        },
    }


def main() -> int:
    args = _build_parser().parse_args()
    ctx = _resolve_context(args)
    primary = _load_candidate(ctx.module, ctx.qa_result_dir, label="QA_optimization.py final")
    comparison = None
    if args.comparison_result_dir is not None:
        comparison = _load_candidate(
            ctx.module,
            args.comparison_result_dir,
            label=str(args.comparison_label),
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{args.stem}.json"
    png_path = args.out_dir / f"{args.stem}.png"
    pdf_path = args.out_dir / f"{args.stem}.pdf"
    summary = _summary(ctx=ctx, primary=primary, comparison=comparison, args=args)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.no_plots:
        _write_figure(
            png_path,
            pdf_path,
            vj=ctx.module,
            primary=primary,
            comparison=comparison,
            theta_points=max(int(args.theta_points), 12),
            zeta_points=max(int(args.zeta_points), 12),
        )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        gate = summary["qa_optimization"]["gate"]
        print(
            "VMEC-backed QA diagnostic written to "
            f"{json_path} (iota={primary.metrics['mean_iota']:.6g}, "
            f"aspect={primary.metrics['aspect_ratio']:.6g}, gate={gate['status']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
