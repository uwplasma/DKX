#!/usr/bin/env python
"""Screen QA/QI NFP candidates for electron-root optimization.

This script is intentionally fast.  It ranks differentiable Boozer-spectrum
proxy candidates and writes a high-fidelity promotion plan; it does not claim a
kinetic SFINCS electron root until the recommended candidate passes real
``sfincs_jax scan-er`` CPU/GPU/Fortran gates.
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
from sfincs_jax.optimization_objectives import (  # noqa: E402
    NeoclassicalObjectiveWeights,
    symmetry_proxy_neoclassical_components,
    symmetry_proxy_neoclassical_objective,
)


def _setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 11.0,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=["qa:2", "qi:1", "qi:2", "qi:3", "qi:4", "qi:5"],
        help="Candidate list as symmetry:nfp, for example qa:2 qi:1 qi:5.",
    )
    parser.add_argument("--steps", type=int, default=70, help="Proxy optimization steps per candidate.")
    parser.add_argument("--learning-rate", type=float, default=0.28, help="Initial gradient-descent step size.")
    parser.add_argument("--n-theta", type=int, default=48, help="Boozer theta grid used by the proxy.")
    parser.add_argument("--n-zeta", type=int, default=44, help="Boozer zeta grid per field period.")
    parser.add_argument("--target-electron-root-drive", type=float, default=0.02)
    parser.add_argument("--target-impurity-flux", type=float, default=0.035)
    parser.add_argument("--max-symmetry-regularization", type=float, default=0.02)
    parser.add_argument("--out-dir", type=Path, default=Path.cwd())
    parser.add_argument("--stem", default="qi_electron_root_nfp_screen")
    parser.add_argument("--no-plots", action="store_true", help="Write JSON only.")
    parser.add_argument("--json", action="store_true", help="Print the JSON summary.")
    return parser


def _parse_candidate(text: str) -> tuple[str, int]:
    try:
        symmetry, nfp_text = str(text).split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"candidate {text!r} must use the form symmetry:nfp"
        ) from exc
    symmetry = symmetry.strip().lower().replace("-", "_")
    if symmetry not in {"qa", "qi"}:
        raise argparse.ArgumentTypeError(f"candidate symmetry must be qa or qi, got {symmetry!r}")
    nfp = int(nfp_text)
    if nfp < 1:
        raise argparse.ArgumentTypeError(f"nfp must be >= 1, got {nfp}")
    return symmetry, nfp


def _default_spectrum(symmetry: str, nfp: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return a compact spectrum with both allowed and penalized content."""

    if symmetry == "qa":
        active = jnp.asarray([0.080, 0.030, -0.024, 0.018, -0.010], dtype=jnp.float64)
        ixm_b = jnp.asarray([0, 1, 1, 2, 2, 3], dtype=jnp.int32)
        ixn_b = jnp.asarray([0, 0, nfp, 0, -nfp, 2 * nfp], dtype=jnp.int32)
        return active, ixm_b, ixn_b
    active = jnp.asarray([0.085, -0.030, 0.014, 0.020, -0.012], dtype=jnp.float64)
    ixm_b = jnp.asarray([0, 0, 0, 1, 1, 2], dtype=jnp.int32)
    ixn_b = jnp.asarray([0, nfp, 2 * nfp, nfp, -nfp, 2 * nfp], dtype=jnp.int32)
    return active, ixm_b, ixn_b


def _weights(symmetry: str) -> NeoclassicalObjectiveWeights:
    if symmetry == "qi":
        return NeoclassicalObjectiveWeights(
            bootstrap=7.0,
            electron_root=750.0,
            main_particle_flux=1.0,
            main_heat_flux=1.0,
            impurity_flux=350.0,
            qa_regularization=6.0,
        )
    return NeoclassicalObjectiveWeights(
        bootstrap=5.0,
        electron_root=500.0,
        main_particle_flux=1.0,
        main_heat_flux=1.0,
        impurity_flux=500.0,
        qa_regularization=0.1,
    )


def _as_float_dict(components: dict[str, Any]) -> dict[str, float]:
    return {key: float(np.asarray(value)) for key, value in components.items()}


def _optimize_candidate(
    *,
    symmetry: str,
    nfp: int,
    steps: int,
    learning_rate: float,
    n_theta: int,
    n_zeta: int,
    target_electron_root_drive: float,
    target_impurity_flux: float,
) -> dict[str, Any]:
    active0, ixm_b, ixn_b = _default_spectrum(symmetry, nfp)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(n_theta), endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / float(nfp), int(n_zeta), endpoint=False, dtype=jnp.float64)
    weights = _weights(symmetry)

    def objective(active: jnp.ndarray) -> jnp.ndarray:
        return symmetry_proxy_neoclassical_objective(
            active,
            ixm_b,
            ixn_b,
            theta=theta,
            zeta=zeta,
            weights=weights,
            symmetry=symmetry,
            nfp=nfp,
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
            symmetry_proxy_neoclassical_components(
                active,
                ixm_b,
                ixn_b,
                theta=theta,
                zeta=zeta,
                symmetry=symmetry,
                nfp=nfp,
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
                lr = min(trial_lr * 1.03, float(learning_rate))
                accepted = True
                break
            trial_lr *= 0.5
        if not accepted:
            active = active - 1.0e-3 * grad
            lr = max(trial_lr, 1.0e-5)

    final = history[-1]
    objective_reduction = history[0]["objective"] - final["objective"]
    return {
        "candidate": f"{symmetry}:nfp{nfp}",
        "symmetry": symmetry,
        "nfp": int(nfp),
        "initial_active_bmnc": [float(v) for v in np.asarray(active0)],
        "final_active_bmnc": [float(v) for v in np.asarray(active)],
        "ixm_b": [int(v) for v in np.asarray(ixm_b)],
        "ixn_b": [int(v) for v in np.asarray(ixn_b)],
        "weights": weights.__dict__,
        "history": history,
        "initial_components": {key: history[0][key] for key in history[0] if key != "step"},
        "final_components": {key: final[key] for key in final if key != "step"},
        "objective_reduction": float(objective_reduction),
        "grid_shape": {"n_theta": int(n_theta), "n_zeta": int(n_zeta)},
    }


def _candidate_gate(
    candidate: dict[str, Any],
    *,
    target_electron_root_drive: float,
    target_impurity_flux: float,
    max_symmetry_regularization: float,
) -> dict[str, Any]:
    final = candidate["final_components"]
    failures: list[str] = []
    if final["electron_root_drive"] < 0.98 * float(target_electron_root_drive):
        failures.append("electron_root_drive below target")
    if final["impurity_outward_proxy"] < 0.98 * float(target_impurity_flux):
        failures.append("impurity_outward_proxy below target")
    if final["symmetry_regularization"] > float(max_symmetry_regularization):
        failures.append("symmetry_regularization above screening cap")
    if candidate["objective_reduction"] < -1.0e-12:
        failures.append("objective increased")
    return {
        "status": "pass" if not failures else "defer",
        "failures": failures,
        "claim_boundary": "proxy screening only; requires high-fidelity sfincs_jax scan-er promotion",
    }


def _selection_score(candidate: dict[str, Any]) -> float:
    final = candidate["final_components"]
    nfp = float(candidate["nfp"])
    if candidate["symmetry"] == "qi":
        # The first kinetic promotion target should stay close to the existing
        # nfp=2 QI fixtures and the electron-root optimization literature unless
        # another NFP has clearly stronger proxy evidence.
        complexity = 0.003 * (nfp - 2.0) ** 2
    else:
        complexity = 0.004 * max(0.0, nfp - 3.0) ** 2
    qa_penalty = 0.01 if candidate["symmetry"] == "qa" else 0.0
    return float(
        final["objective"]
        - 2.0 * final["electron_root_drive"]
        - 0.5 * final["impurity_outward_proxy"]
        + 3.0 * final["symmetry_regularization"]
        + complexity
        + qa_penalty
    )


def _choose_recommended_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the next high-fidelity promotion target.

    Passing proxy gates always win.  If no candidate passes and the QA
    candidates are deferred, pivot to the best QI candidate rather than
    re-running a weak QA proxy.  This keeps the public workflow aligned with the
    current evidence: QA has low/mid-resolution positive-root artifacts, while a
    QI electron-root lane still needs its first kinetic promotion artifact.
    """

    passing = [row for row in candidates if row["screening_gate"]["status"] == "pass"]
    if passing:
        return min(passing, key=lambda row: row["selection_score"])
    qi_rows = [row for row in candidates if row["symmetry"] == "qi"]
    if qi_rows:
        return min(qi_rows, key=lambda row: row["selection_score"])
    return min(candidates, key=lambda row: row["selection_score"])


def _promotion_plan(best: dict[str, Any]) -> dict[str, Any]:
    return {
        "recommended_candidate": best["candidate"],
        "next_commands": [
            "generate or select a VMEC/booz_xform_jax equilibrium matching the candidate symmetry and nfp",
            "run sfincs_jax scan-er on the accepted radius/profile with a positive-Er bracket",
            "audit with examples/optimization/evaluate_sfincs_jax_promotion_scan.py --require-electron-root",
            "repeat on CPU and GPU and compare with examples/optimization/compare_sfincs_jax_promotion_runs.py",
        ],
        "required_gates": [
            "positive ambipolar root bracket from completed kinetic scan",
            "linear residual below target at all scan points",
            "CPU/GPU agreement for selected root and transport objectives",
            "resolution ladder with root drift below the documented tolerance",
            "Fortran-v3 comparison when the candidate is representable in shared inputs",
        ],
    }


def _bhat(candidate: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    active = jnp.asarray(candidate["final_active_bmnc"], dtype=jnp.float64)
    coeff = jnp.concatenate([jnp.asarray([1.0], dtype=active.dtype), active])
    ixm_b = jnp.asarray(candidate["ixm_b"], dtype=jnp.int32)
    ixn_b = jnp.asarray(candidate["ixn_b"], dtype=jnp.int32)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 64, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / float(candidate["nfp"]), 56, endpoint=False, dtype=jnp.float64)
    bhat = boozer_bhat_from_spectrum(theta, zeta, bmnc_b=coeff, ixm_b=ixm_b, ixn_b=ixn_b, normalize=True)
    return np.asarray(theta), np.asarray(zeta), np.asarray(bhat)


def _write_figure(path_png: Path, path_pdf: Path, *, summary: dict[str, Any]) -> None:
    _setup_mpl()
    candidates = summary["candidates"]
    labels = [row["candidate"] for row in candidates]
    scores = np.asarray([row["selection_score"] for row in candidates], dtype=float)
    root_drive = np.asarray([row["final_components"]["electron_root_drive"] for row in candidates], dtype=float)
    symmetry_reg = np.asarray([row["final_components"]["symmetry_regularization"] for row in candidates], dtype=float)
    best = summary["recommended_candidate"]
    theta, zeta, bhat = _bhat(best)

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.8), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0, 0].bar(x, scores, color="#4c78a8")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0, 0].set_title("A. Proxy selection score")
    axes[0, 0].set_ylabel("lower is better")

    axes[0, 1].bar(x, root_drive, color="#f58518", label="electron-root drive")
    axes[0, 1].axhline(summary["targets"]["electron_root_drive"], color="k", ls="--", lw=1.0)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels, rotation=35, ha="right")
    axes[0, 1].set_title("B. Electron-root proxy")
    axes[0, 1].set_ylabel("proxy amplitude")

    axes[1, 0].bar(x, symmetry_reg, color="#54a24b")
    axes[1, 0].axhline(summary["targets"]["max_symmetry_regularization"], color="k", ls="--", lw=1.0)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels, rotation=35, ha="right")
    axes[1, 0].set_title("C. Symmetry regularization")
    axes[1, 0].set_ylabel("penalty")

    c = axes[1, 1].contourf(zeta, theta, bhat, levels=24, cmap="jet")
    axes[1, 1].set_title(f"D. Recommended {best['candidate']} $B/B_{{00}}$")
    axes[1, 1].set_xlabel(r"$\zeta$")
    axes[1, 1].set_ylabel(r"$\theta$")
    fig.colorbar(c, ax=axes[1, 1], shrink=0.9)

    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    candidate_specs = [_parse_candidate(text) for text in args.candidates]
    results = [
        _optimize_candidate(
            symmetry=symmetry,
            nfp=nfp,
            steps=int(args.steps),
            learning_rate=float(args.learning_rate),
            n_theta=int(args.n_theta),
            n_zeta=int(args.n_zeta),
            target_electron_root_drive=float(args.target_electron_root_drive),
            target_impurity_flux=float(args.target_impurity_flux),
        )
        for symmetry, nfp in candidate_specs
    ]
    for row in results:
        row["screening_gate"] = _candidate_gate(
            row,
            target_electron_root_drive=float(args.target_electron_root_drive),
            target_impurity_flux=float(args.target_impurity_flux),
            max_symmetry_regularization=float(args.max_symmetry_regularization),
        )
        row["selection_score"] = _selection_score(row)
    best = _choose_recommended_candidate(results)
    summary = {
        "workflow": "sfincs_jax_qi_qa_electron_root_nfp_screening_proxy",
        "claim_boundary": "proxy screening only; not a kinetic SFINCS transport claim",
        "targets": {
            "electron_root_drive": float(args.target_electron_root_drive),
            "impurity_outward_proxy": float(args.target_impurity_flux),
            "max_symmetry_regularization": float(args.max_symmetry_regularization),
        },
        "candidates": results,
        "recommended_candidate": best,
        "promotion_plan": _promotion_plan(best),
    }

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.stem}.json"
    png_path = out_dir / f"{args.stem}.png"
    pdf_path = out_dir / f"{args.stem}.pdf"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.no_plots:
        _write_figure(png_path, pdf_path, summary=summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print("QA/QI electron-root NFP screening complete")
        print(f"  recommended candidate: {best['candidate']}")
        print(f"  gate: {best['screening_gate']['status']}")
        print(f"  summary: {json_path}")
        if not args.no_plots:
            print(f"  figure:  {png_path}")
            print(f"  pdf:     {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
