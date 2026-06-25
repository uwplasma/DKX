#!/usr/bin/env python
"""QA nfp=2 optimization lane with sfincs_jax neoclassical objectives.

This example demonstrates the recommended two-tier architecture:

1. A cheap, differentiable Boozer-spectrum proxy objective is used inside the
   optimizer loop.
2. Accepted designs are promoted to high-fidelity ``sfincs_jax`` kinetic gates
   only after they satisfy the proxy/autodiff checks.

The script is intentionally fast and self-contained.  It does not claim that
the proxy objective is a kinetic SFINCS solve; the output JSON records the
promotion gates required before using a design in a publication claim.
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

from sfincs_jax.jax_geometry_adapters import boozer_bhat_from_spectrum  # noqa: E402
from sfincs_jax.workflows.optimization_objectives import (  # noqa: E402
    NeoclassicalObjectiveWeights,
    qa_proxy_gradient_gate,
    qa_proxy_neoclassical_components,
    qa_proxy_neoclassical_objective,
)


def _setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10.2,
            "axes.labelsize": 10.2,
            "axes.titlesize": 11.0,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.2,
            "ytick.labelsize": 9.2,
            "axes.grid": True,
            "grid.alpha": 0.23,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=80, help="JAX proxy-optimization steps.")
    parser.add_argument("--learning-rate", type=float, default=0.35, help="Initial gradient-descent step size.")
    parser.add_argument(
        "--objective",
        choices=("bootstrap", "electron-root", "flux-selective", "balanced"),
        default="balanced",
        help="Preset for the proxy objective weights.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path.cwd(), help="Directory for PNG/PDF/JSON outputs.")
    parser.add_argument("--stem", default="qa_nfp2_sfincs_jax_optimization_lane", help="Output file stem.")
    parser.add_argument("--no-plots", action="store_true", help="Write JSON only.")
    parser.add_argument("--json", action="store_true", help="Print the machine-readable summary.")
    return parser


def _objective_weights(kind: str) -> NeoclassicalObjectiveWeights:
    if kind == "bootstrap":
        return NeoclassicalObjectiveWeights(
            bootstrap=20.0,
            electron_root=0.0,
            main_particle_flux=1.0,
            main_heat_flux=1.0,
            impurity_flux=0.0,
            qa_regularization=4.0,
        )
    if kind == "electron-root":
        return NeoclassicalObjectiveWeights(
            bootstrap=6.0,
            electron_root=600.0,
            main_particle_flux=1.0,
            main_heat_flux=1.0,
            impurity_flux=120.0,
            qa_regularization=0.1,
        )
    if kind == "flux-selective":
        return NeoclassicalObjectiveWeights(
            bootstrap=4.0,
            electron_root=120.0,
            main_particle_flux=2.0,
            main_heat_flux=2.0,
            impurity_flux=800.0,
            qa_regularization=0.1,
        )
    if kind == "balanced":
        return NeoclassicalObjectiveWeights(
            bootstrap=5.0,
            electron_root=500.0,
            main_particle_flux=1.0,
            main_heat_flux=1.0,
            impurity_flux=500.0,
            qa_regularization=0.1,
        )
    raise ValueError(f"unknown objective preset: {kind}")


def _default_spectrum() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a small nfp=2 QA-like Boozer spectrum.

    The first mode is the fixed ``B00`` term.  Active parameters begin at index
    one and include both QA-like ``n=0`` shaping and non-QA ``n != 0`` ripples.
    """

    active = jnp.asarray([0.080, 0.030, -0.024, 0.018, -0.010], dtype=jnp.float64)
    ixm_b = jnp.asarray([0, 1, 1, 2, 2, 3], dtype=jnp.int32)
    ixn_b = jnp.asarray([0, 0, 2, 0, -2, 4], dtype=jnp.int32)
    return active, ixm_b, ixn_b


def _as_float_dict(components: dict[str, Any]) -> dict[str, float]:
    return {key: float(np.asarray(value)) for key, value in components.items()}


def _bhat_grid(active: jnp.ndarray, ixm_b: jnp.ndarray, ixn_b: jnp.ndarray, theta, zeta) -> np.ndarray:
    coeff = jnp.concatenate([jnp.asarray([1.0], dtype=active.dtype), active])
    return np.asarray(
        boozer_bhat_from_spectrum(
            theta,
            zeta,
            bmnc_b=coeff,
            ixm_b=ixm_b,
            ixn_b=ixn_b,
            normalize=True,
        )
    )


def _optimize(
    active0: jnp.ndarray,
    ixm_b: jnp.ndarray,
    ixn_b: jnp.ndarray,
    *,
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
    steps: int,
    learning_rate: float,
    weights: NeoclassicalObjectiveWeights,
    target_electron_root_drive: float,
    target_impurity_flux: float,
) -> dict[str, Any]:
    def objective(active):
        return qa_proxy_neoclassical_objective(
            active,
            ixm_b,
            ixn_b,
            theta=theta,
            zeta=zeta,
            weights=weights,
            target_electron_root_drive=target_electron_root_drive,
            target_impurity_flux=target_impurity_flux,
        )

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    active = active0
    lr = float(learning_rate)
    history: list[dict[str, float]] = []

    for step in range(int(steps) + 1):
        value, grad = value_and_grad(active)
        components = _as_float_dict(
            qa_proxy_neoclassical_components(
                active,
                ixm_b,
                ixn_b,
                theta=theta,
                zeta=zeta,
                target_electron_root_drive=target_electron_root_drive,
                target_impurity_flux=target_impurity_flux,
            )
        )
        history.append(
            {
                "step": float(step),
                "objective": float(value),
                "gradient_norm": float(jnp.linalg.norm(grad)),
                "learning_rate": float(lr),
                **components,
            }
        )
        if step == int(steps):
            break

        accepted = False
        trial_lr = lr
        for _ in range(10):
            candidate = active - trial_lr * grad
            candidate_value = float(objective(candidate))
            if np.isfinite(candidate_value) and candidate_value <= float(value):
                active = candidate
                lr = min(trial_lr * 1.04, float(learning_rate))
                accepted = True
                break
            trial_lr *= 0.5
        if not accepted:
            active = active - 1.0e-3 * grad
            lr = max(trial_lr, 1.0e-5)

    return {
        "initial_active_bmnc": [float(x) for x in np.asarray(active0)],
        "final_active_bmnc": [float(x) for x in np.asarray(active)],
        "ixm_b": [int(x) for x in np.asarray(ixm_b)],
        "ixn_b": [int(x) for x in np.asarray(ixn_b)],
        "history": history,
        "initial_components": {key: history[0][key] for key in history[0] if key not in {"step"}},
        "final_components": {key: history[-1][key] for key in history[-1] if key not in {"step"}},
    }


def _write_figure(
    path_png: Path,
    path_pdf: Path,
    *,
    result: dict[str, Any],
    theta: jnp.ndarray,
    zeta: jnp.ndarray,
) -> None:
    _setup_mpl()
    history = result["history"]
    steps = np.asarray([row["step"] for row in history], dtype=float)
    objective = np.asarray([row["objective"] for row in history], dtype=float)
    grad_norm = np.asarray([row["gradient_norm"] for row in history], dtype=float)

    active0 = jnp.asarray(result["initial_active_bmnc"], dtype=jnp.float64)
    active1 = jnp.asarray(result["final_active_bmnc"], dtype=jnp.float64)
    ixm_b = jnp.asarray(result["ixm_b"], dtype=jnp.int32)
    ixn_b = jnp.asarray(result["ixn_b"], dtype=jnp.int32)
    bhat0 = _bhat_grid(active0, ixm_b, ixn_b, theta, zeta)
    bhat1 = _bhat_grid(active1, ixm_b, ixn_b, theta, zeta)
    contour_levels = np.linspace(float(min(bhat0.min(), bhat1.min())), float(max(bhat0.max(), bhat1.max())), 22)

    component_names = [
        "bootstrap",
        "electron_root",
        "main_particle_flux",
        "main_heat_flux",
        "impurity_flux",
        "qa_regularization",
    ]
    labels = [
        "bootstrap",
        "electron root",
        "main particle",
        "main heat",
        "impurity target",
        "non-QA",
    ]
    initial = np.asarray([result["initial_components"][name] for name in component_names], dtype=float)
    final = np.asarray([result["final_components"][name] for name in component_names], dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.4), constrained_layout=True)
    ax = axes[0, 0]
    ax.semilogy(steps, objective, color="#0f4c81", lw=2.4)
    ax.set_title("A. Differentiable proxy objective")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("weighted objective")

    ax = axes[0, 1]
    ax.semilogy(steps, grad_norm, color="#d95f02", lw=2.4)
    ax.set_title("B. JAX gradient norm")
    ax.set_xlabel("optimization step")
    ax.set_ylabel(r"$||\nabla J||_2$")

    ax = axes[0, 2]
    x = np.arange(len(component_names))
    width = 0.38
    ax.bar(x - width / 2, initial, width=width, label="initial", color="#8da0cb")
    ax.bar(x + width / 2, final, width=width, label="optimized", color="#66c2a5")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_title("C. Objective terms")
    ax.set_ylabel("proxy penalty")
    ax.legend(loc="best")

    zeta_np = np.asarray(zeta)
    theta_np = np.asarray(theta)
    ax = axes[1, 0]
    c0 = ax.contourf(zeta_np, theta_np, bhat0, levels=contour_levels, cmap="jet")
    ax.set_title(r"D. Initial $B/B_{00}$")
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$\theta$")
    fig.colorbar(c0, ax=ax, shrink=0.88)

    ax = axes[1, 1]
    c1 = ax.contourf(zeta_np, theta_np, bhat1, levels=contour_levels, cmap="jet")
    ax.set_title(r"E. Optimized $B/B_{00}$")
    ax.set_xlabel(r"$\zeta$")
    ax.set_ylabel(r"$\theta$")
    fig.colorbar(c1, ax=ax, shrink=0.88)

    ax = axes[1, 2]
    targets = result["proxy_targets"]
    ax.plot(steps, [row["electron_root_drive"] for row in history], label="electron-root drive", color="#0f4c81")
    ax.plot(steps, [row["impurity_outward_proxy"] for row in history], label="impurity outward proxy", color="#1b9e77")
    ax.axhline(float(targets["electron_root_drive"]), color="#0f4c81", ls="--", alpha=0.45)
    ax.axhline(float(targets["impurity_outward_proxy"]), color="#1b9e77", ls="--", alpha=0.45)
    ax.set_title("F. Promotion-target proxies")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("proxy amplitude")
    ax.legend(loc="best")

    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)


def _promotion_plan() -> dict[str, Any]:
    return {
        "claim_boundary": (
            "This run optimizes a differentiable proxy. Publication claims require "
            "accepted designs to pass high-fidelity sfincs_jax kinetic gates."
        ),
        "required_high_fidelity_gates": [
            "same-profile SFINCS_JAX Er scan over each selected radius",
            "ambipolar root bracketing with a resolved positive electron root when requested",
            "bootstrap-current normalization audit using FSABjHatOverRootFSAB2",
            "particle/heat/impurity flux sign-convention audit",
            "linear residual convergence and solver-path provenance",
            "CPU/GPU agreement for selected final designs",
            "SFINCS Fortran v3 comparison when the input is in the shared model scope",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    jax.config.update("jax_enable_x64", True)

    active0, ixm_b, ixn_b = _default_spectrum()
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 48, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, jnp.pi, 40, endpoint=False, dtype=jnp.float64)
    weights = _objective_weights(args.objective)
    target_electron_root_drive = 0.02
    target_impurity_flux = 0.035
    result = _optimize(
        active0,
        ixm_b,
        ixn_b,
        theta=theta,
        zeta=zeta,
        steps=int(args.steps),
        learning_rate=float(args.learning_rate),
        weights=weights,
        target_electron_root_drive=target_electron_root_drive,
        target_impurity_flux=target_impurity_flux,
    )
    result["workflow"] = "qa_nfp2_sfincs_jax_neoclassical_optimization_proxy"
    result["nfp"] = 2
    result["objective_preset"] = args.objective
    result["weights"] = weights.__dict__
    result["proxy_targets"] = {
        "electron_root_drive": target_electron_root_drive,
        "impurity_outward_proxy": target_impurity_flux,
    }
    result["autodiff_gradient_gate"] = qa_proxy_gradient_gate()
    result["promotion_plan"] = _promotion_plan()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.stem}.json"
    png_path = out_dir / f"{args.stem}.png"
    pdf_path = out_dir / f"{args.stem}.pdf"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.no_plots:
        _write_figure(png_path, pdf_path, result=result, theta=theta, zeta=zeta)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        initial = result["history"][0]["objective"]
        final = result["history"][-1]["objective"]
        print("QA nfp=2 sfincs_jax optimization proxy complete")
        print(f"  objective: {initial:.6e} -> {final:.6e}")
        print(f"  gradient gate: {result['autodiff_gradient_gate']['status']}")
        print(f"  summary: {json_path}")
        if not args.no_plots:
            print(f"  figure:  {png_path}")
            print(f"  pdf:     {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
