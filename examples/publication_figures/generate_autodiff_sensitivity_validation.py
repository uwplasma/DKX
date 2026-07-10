#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("This example requires matplotlib. Install with: pip install matplotlib") from exc

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.geometry import boozer_geometry_scheme4
from sfincs_jax.solvers.implicit import linear_custom_solve
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.validation.artifacts import (
    PAUL_2019_ADJOINT_URL,
    SFINCS_ADJOINT_APS_URL,
    build_autodiff_sensitivity_validation_summary,
    load_autodiff_sensitivity_summary,
)
from sfincs_jax.operators.profile_system import (
    apply_v3_full_system_operator,
    full_system_operator_from_namelist,
    rhs_v3_full_system,
)


DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "examples" / "publication_figures" / "artifacts"
DEFAULT_OUT_DIR = _REPO_ROOT / "docs" / "_static" / "figures" / "paper"
DEFAULT_SUMMARY = DEFAULT_ARTIFACT_DIR / "sfincs_jax_autodiff_sensitivity_validation_summary.json"
DEFAULT_GRADIENT_STEM = "sfincs_jax_autodiff_gradient_check"
DEFAULT_SENSITIVITY_STEM = "sfincs_jax_autodiff_sensitivity_map"
DEFAULT_INPUT = _REPO_ROOT / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"


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
            "grid.alpha": 0.24,
            "grid.linestyle": "-",
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_autodiff_sensitivity_validation",
        description="Generate publication-grade autodiff and sensitivity validation artifacts.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Tiny full-system PAS fixture input.")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gradient-stem", default=DEFAULT_GRADIENT_STEM)
    parser.add_argument("--sensitivity-stem", default=DEFAULT_SENSITIVITY_STEM)
    parser.add_argument("--plot-only", action="store_true", help="Reuse --summary-json and only regenerate figures.")
    parser.add_argument("--compute-only", action="store_true", help="Write summary JSON but skip figure generation.")
    parser.add_argument("--shift", type=float, default=1.0e-3, help="Operator shift p for the SFINCS solve check.")
    parser.add_argument(
        "--fd-eps",
        default="1e-4,3e-5,1e-5",
        help="Comma-separated finite-difference steps for the SFINCS gradient sweep.",
    )
    parser.add_argument("--n-theta", type=int, default=25, help="Theta grid for the geometry sensitivity map.")
    parser.add_argument("--n-zeta", type=int, default=25, help="Zeta grid for the geometry sensitivity map.")
    return parser


def _rel_error(a: float, b: float) -> float:
    return float(abs(float(a) - float(b)) / max(abs(float(a)), abs(float(b)), np.finfo(float).tiny))


def _dense_problem_check(*, solver: str) -> dict[str, object]:
    a0 = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 3.0, 1.0, 0.0],
            [0.0, 1.0, 2.5, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([1.0, 2.0, -1.0, 0.5], dtype=jnp.float64)
    p0 = jnp.asarray(0.2, dtype=jnp.float64)

    def solve_at(p: jnp.ndarray):
        a = a0 + jnp.asarray(p, dtype=jnp.float64) * jnp.eye(4, dtype=jnp.float64)

        def mv(x: jnp.ndarray) -> jnp.ndarray:
            return a @ x

        return linear_custom_solve(matvec=mv, b=b, tol=1e-12, maxiter=100, solver=solver, solver_jit=False)

    def objective(p: jnp.ndarray) -> jnp.ndarray:
        x = solve_at(p).x
        return 0.5 * jnp.vdot(x, x)

    t0 = time.perf_counter()
    value = float(objective(p0))
    grad = float(jax.grad(objective)(p0))
    gradient_runtime = time.perf_counter() - t0
    eps = 1.0e-6
    fd = (float(objective(p0 + eps)) - float(objective(p0 - eps))) / (2.0 * eps)
    result = solve_at(p0)

    a = a0 + p0 * jnp.eye(4, dtype=jnp.float64)
    adjoint = linear_custom_solve(
        matvec=lambda y: a.T @ y,
        b=result.x,
        tol=1e-12,
        maxiter=100,
        solver=solver,
        solver_jit=False,
    )
    adjoint_residual = float(jnp.linalg.norm(a.T @ adjoint.x - result.x))
    return {
        "name": f"dense_4x4_{solver}",
        "kind": "well_conditioned_custom_linear_solve",
        "solver": solver,
        "parameter": "diagonal_shift",
        "parameter_value": float(p0),
        "objective": value,
        "autodiff_gradient": grad,
        "finite_difference_gradient": float(fd),
        "absolute_error": float(abs(grad - fd)),
        "relative_error": _rel_error(grad, fd),
        "finite_difference_eps": eps,
        "primal_residual_norm": float(result.residual_norm),
        "adjoint_residual_norm": adjoint_residual,
        "gradient_runtime_s": float(gradient_runtime),
    }


def _sfincs_shift_checks(*, input_path: Path, shift: float, fd_eps: list[float]) -> tuple[dict[str, object], list[dict[str, object]]]:
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    b = rhs_v3_full_system(op)
    p0 = jnp.asarray(float(shift), dtype=jnp.float64)

    def solve_at(p: jnp.ndarray):
        p = jnp.asarray(p, dtype=jnp.float64)

        def mv(x: jnp.ndarray) -> jnp.ndarray:
            return apply_v3_full_system_operator(op, x) + p * x

        return linear_custom_solve(
            matvec=mv,
            b=b,
            tol=1e-11,
            restart=50,
            maxiter=300,
            solver="gmres",
            size_hint=op.total_size,
            solver_jit=False,
        )

    def objective(p: jnp.ndarray) -> jnp.ndarray:
        x = solve_at(p).x
        return 0.5 * jnp.vdot(x, x)

    t0 = time.perf_counter()
    value = float(objective(p0))
    grad = float(jax.grad(objective)(p0))
    gradient_runtime = time.perf_counter() - t0

    sweep: list[dict[str, object]] = []
    for eps in fd_eps:
        eps = float(eps)
        fd = (float(objective(p0 + eps)) - float(objective(p0 - eps))) / (2.0 * eps)
        sweep.append(
            {
                "eps": eps,
                "finite_difference_gradient": float(fd),
                "autodiff_gradient": grad,
                "absolute_error": float(abs(grad - fd)),
                "relative_error": _rel_error(grad, fd),
            }
        )
    best = min(sweep, key=lambda row: float(row["relative_error"]))
    result = solve_at(p0)

    def mv0(x: jnp.ndarray) -> jnp.ndarray:
        return apply_v3_full_system_operator(op, x) + p0 * x

    zeros = jnp.zeros_like(result.x)

    def mv0_t(y: jnp.ndarray) -> jnp.ndarray:
        return jax.linear_transpose(mv0, zeros)(y)[0]

    adjoint = linear_custom_solve(
        matvec=mv0_t,
        b=result.x,
        tol=1e-11,
        restart=50,
        maxiter=300,
        solver="gmres",
        size_hint=op.total_size,
        solver_jit=False,
    )
    adjoint_residual = float(jnp.linalg.norm(mv0_t(adjoint.x) - result.x))
    check = {
        "name": "sfincs_full_system_shift_gmres",
        "kind": "sfincs_full_system_custom_linear_solve",
        "solver": "gmres",
        "input": str(input_path.relative_to(_REPO_ROOT)),
        "parameter": "operator_identity_shift",
        "parameter_value": float(p0),
        "system_size": int(op.total_size),
        "objective": value,
        "autodiff_gradient": grad,
        "finite_difference_gradient": float(best["finite_difference_gradient"]),
        "absolute_error": float(best["absolute_error"]),
        "relative_error": float(best["relative_error"]),
        "finite_difference_eps": float(best["eps"]),
        "primal_residual_norm": float(result.residual_norm),
        "adjoint_residual_norm": adjoint_residual,
        "gradient_runtime_s": float(gradient_runtime),
    }
    return check, sweep


def _geometry_sensitivity(*, n_theta: int, n_zeta: int) -> dict[str, object]:
    theta = jnp.linspace(0.0, 2 * jnp.pi, int(n_theta), endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, 2 * jnp.pi / 5, int(n_zeta), endpoint=False, dtype=jnp.float64)
    amps0 = jnp.asarray([0.04645, -0.04351, -0.01902], dtype=jnp.float64)
    harmonic_labels = ["(m=0,n=1)", "(m=1,n=1)", "(m=1,n=0)"]

    def b_field(amps: jnp.ndarray) -> jnp.ndarray:
        return boozer_geometry_scheme4(theta=theta, zeta=zeta, harmonics_amp0=amps).b_hat

    def objective(amps: jnp.ndarray) -> jnp.ndarray:
        b_hat = b_field(amps)
        return jnp.mean(b_hat**2)

    t0 = time.perf_counter()
    maps = jax.jacrev(b_field)(amps0)  # (theta, zeta, harmonic)
    grad = jax.grad(objective)(amps0)
    gradient_runtime = time.perf_counter() - t0
    eps = 1.0e-5
    fd_grad = []
    for i in range(int(amps0.size)):
        step = jnp.zeros_like(amps0).at[i].set(eps)
        fd_grad.append(float((objective(amps0 + step) - objective(amps0 - step)) / (2.0 * eps)))
    fd_grad_arr = np.asarray(fd_grad, dtype=np.float64)
    grad_arr = np.asarray(grad, dtype=np.float64)
    map_arr = np.moveaxis(np.asarray(maps, dtype=np.float64), -1, 0)
    map_stats = []
    for label, arr in zip(harmonic_labels, map_arr, strict=True):
        map_stats.append(
            {
                "harmonic": label,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "rms": float(np.sqrt(np.mean(arr**2))),
            }
        )
    return {
        "kind": "scheme4_boozer_harmonic_map",
        "objective": "mean(BHat**2)",
        "n_theta": int(n_theta),
        "n_zeta": int(n_zeta),
        "harmonic_labels": harmonic_labels,
        "amplitudes": [float(v) for v in np.asarray(amps0, dtype=np.float64)],
        "objective_value": float(objective(amps0)),
        "autodiff_gradient": [float(v) for v in grad_arr],
        "finite_difference_gradient": [float(v) for v in fd_grad_arr],
        "gradient_relative_error": _rel_error(float(np.linalg.norm(grad_arr)), float(np.linalg.norm(fd_grad_arr))),
        "finite_difference_eps": eps,
        "gradient_runtime_s": float(gradient_runtime),
        "map_stats": map_stats,
        "maps_dBhat_damp": map_arr.tolist(),
    }


def _cost_scaling(*, measured_gradient_runtime_s: float, measured_objective_runtime_s: float) -> list[dict[str, object]]:
    rows = []
    for n_params in (1, 3, 10, 30, 100):
        rows.append(
            {
                "n_parameters": int(n_params),
                "implicit_solve_count_model": 2,
                "centered_finite_difference_solve_count_model": int(2 * n_params),
                "measured_single_gradient_runtime_s": float(measured_gradient_runtime_s),
                "measured_single_objective_runtime_s": float(measured_objective_runtime_s),
                "projected_finite_difference_runtime_s": float(2 * n_params * measured_objective_runtime_s),
            }
        )
    return rows


def build_summary(*, input_path: Path, shift: float, fd_eps: list[float], n_theta: int, n_zeta: int) -> dict[str, object]:
    dense_gmres = _dense_problem_check(solver="gmres")
    dense_bicgstab = _dense_problem_check(solver="bicgstab")
    sfincs_check, sweep = _sfincs_shift_checks(input_path=input_path, shift=float(shift), fd_eps=fd_eps)
    geometry = _geometry_sensitivity(n_theta=int(n_theta), n_zeta=int(n_zeta))
    measured_obj_runtime = float(sfincs_check["gradient_runtime_s"]) / 3.0
    metadata = {
        "schema_version": 1,
        "kind": "autodiff_sensitivity_validation",
        "source_script": str(Path(__file__).resolve().relative_to(_REPO_ROOT)),
        "literature": [PAUL_2019_ADJOINT_URL, SFINCS_ADJOINT_APS_URL],
        "jax_backend": str(jax.default_backend()),
        "jax_devices": [str(device) for device in jax.devices()],
        "scope": (
            "Bounded manuscript artifact: validates implicit differentiation through a pinned "
            "SFINCS full-system linear solve and differentiable scheme-4 Boozer harmonic maps. "
            "It does not claim full VMEC-boundary optimization."
        ),
    }
    return build_autodiff_sensitivity_validation_summary(
        gradient_checks=[dense_gmres, dense_bicgstab, sfincs_check],
        finite_difference_sweep=sweep,
        geometry_sensitivity=geometry,
        cost_scaling=_cost_scaling(
            measured_gradient_runtime_s=float(sfincs_check["gradient_runtime_s"]),
            measured_objective_runtime_s=measured_obj_runtime,
        ),
        metadata=metadata,
        relative_error_gate=1.0e-4,
        residual_gate=1.0e-8,
    )


def write_summary_json(*, summary_path: Path, payload: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def plot_gradient_check(*, payload: dict[str, object], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = list(payload["gradient_checks"])
    sweep = list(payload["finite_difference_sweep"])
    cost = list(payload["cost_scaling"])
    short_names = {
        "dense_4x4_gmres": "dense GMRES",
        "dense_4x4_bicgstab": "dense BiCGStab",
        "sfincs_full_system_shift_gmres": "SFINCS full-system GMRES",
    }

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.6), constrained_layout=True)

    fd = np.asarray([float(row["finite_difference_gradient"]) for row in checks], dtype=np.float64)
    ad = np.asarray([float(row["autodiff_gradient"]) for row in checks], dtype=np.float64)
    labels = [short_names.get(str(row["name"]), str(row["name"])) for row in checks]
    lim = float(np.max(np.abs(np.concatenate([fd, ad])))) * 1.15
    axes[0, 0].plot([-lim, lim], [-lim, lim], color="black", linewidth=1.0, alpha=0.65)
    colors = ["#0f4c81", "#1b9e77", "#d95f02"]
    markers = ["o", "s", "^"]
    for x, y, label, color, marker in zip(fd, ad, labels, colors, markers, strict=True):
        axes[0, 0].scatter([x], [y], s=58, color=color, marker=marker, label=label)
    axes[0, 0].set_title("A. Gradient parity")
    axes[0, 0].set_xlabel("centered finite difference")
    axes[0, 0].set_ylabel("autodiff / implicit diff")
    axes[0, 0].set_xlim(-lim, lim)
    axes[0, 0].set_ylim(-lim, lim)
    axes[0, 0].legend(loc="upper left")

    eps = np.asarray([float(row["eps"]) for row in sweep], dtype=np.float64)
    rel = np.asarray([float(row["relative_error"]) for row in sweep], dtype=np.float64)
    axes[0, 1].loglog(eps, rel, marker="o", color="#d95f02")
    axes[0, 1].axhline(float(payload["gates"]["relative_error_gate"]), color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].invert_xaxis()
    axes[0, 1].set_title("B. Finite-difference step sweep")
    axes[0, 1].set_xlabel(r"step $\epsilon$")
    axes[0, 1].set_ylabel("relative gradient error")

    x = np.arange(len(checks))
    primal = np.asarray([float(row["primal_residual_norm"]) for row in checks], dtype=np.float64)
    adjoint = np.asarray([float(row["adjoint_residual_norm"]) for row in checks], dtype=np.float64)
    width = 0.36
    axes[1, 0].bar(x - width / 2, np.maximum(primal, np.finfo(float).tiny), width, label="primal")
    axes[1, 0].bar(x + width / 2, np.maximum(adjoint, np.finfo(float).tiny), width, label="adjoint")
    axes[1, 0].set_yscale("log")
    axes[1, 0].axhline(float(payload["gates"]["residual_gate"]), color="black", linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("C. Solve residual gates")
    axes[1, 0].set_xticks(x, labels, rotation=12, ha="right")
    axes[1, 0].set_ylabel(r"$\|Ax-b\|_2$")
    axes[1, 0].legend(loc="best")

    n_params = np.asarray([int(row["n_parameters"]) for row in cost], dtype=np.int64)
    implicit = np.asarray([int(row["implicit_solve_count_model"]) for row in cost], dtype=np.int64)
    finite_diff = np.asarray([int(row["centered_finite_difference_solve_count_model"]) for row in cost], dtype=np.int64)
    axes[1, 1].plot(n_params, implicit, marker="o", label="implicit diff", color="#1b9e77")
    axes[1, 1].plot(n_params, finite_diff, marker="s", label="centered finite diff", color="#d95f02")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("D. Gradient solve-count scaling")
    axes[1, 1].set_xlabel("number of parameters")
    axes[1, 1].set_ylabel("linear solves per gradient")
    axes[1, 1].legend(loc="best")

    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def plot_sensitivity_map(*, payload: dict[str, object], out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    geom = dict(payload["geometry_sensitivity"])
    maps = np.asarray(geom["maps_dBhat_damp"], dtype=np.float64)
    labels = list(geom["harmonic_labels"])
    vmax = float(np.max(np.abs(maps)))
    fig, axes = plt.subplots(1, maps.shape[0], figsize=(12.0, 3.8), constrained_layout=True)
    if maps.shape[0] == 1:
        axes = [axes]
    for ax, arr, label in zip(axes, maps, labels, strict=True):
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(rf"$\partial \hat B / \partial a$ {label}")
        ax.set_xlabel(r"$\zeta$ index")
        ax.set_ylabel(r"$\theta$ index")
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.86)
    cbar.set_label(r"$\partial \hat B / \partial a$")
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    _setup_mpl()
    args = _build_parser().parse_args(argv)
    summary_json = Path(args.summary_json)
    if args.plot_only:
        payload = dict(load_autodiff_sensitivity_summary(summary_json))
    else:
        fd_eps = [float(v.strip()) for v in str(args.fd_eps).split(",") if v.strip()]
        payload = build_summary(
            input_path=Path(args.input),
            shift=float(args.shift),
            fd_eps=fd_eps,
            n_theta=int(args.n_theta),
            n_zeta=int(args.n_zeta),
        )
        write_summary_json(summary_path=summary_json, payload=payload)

    if not args.compute_only:
        plot_gradient_check(payload=payload, out_dir=Path(args.out_dir), stem=str(args.gradient_stem))
        plot_sensitivity_map(payload=payload, out_dir=Path(args.out_dir), stem=str(args.sensitivity_stem))

    print(f"Wrote autodiff validation summary to {summary_json}")
    if not args.compute_only:
        print(f"Wrote gradient-check figure to {Path(args.out_dir) / (str(args.gradient_stem) + '.png')}")
        print(f"Wrote sensitivity-map figure to {Path(args.out_dir) / (str(args.sensitivity_stem) + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
