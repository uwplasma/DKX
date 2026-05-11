#!/usr/bin/env python
# ruff: noqa: E402
"""Generate the W7-X high-nu performance and residual-gate figure.

The figure is intentionally built from a small, checked summary rather than
rerunning the full W7-X point. The full point is expensive enough that the
reproducible command and residual diagnostics belong in docs, while CI should
only validate the artifact math and plotting path.
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("This example requires matplotlib. Install sfincs_jax first.") from exc

import numpy as np


DEFAULT_OUT_DIR = _REPO_ROOT / "docs" / "_static" / "figures" / "paper"
DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "examples" / "publication_figures" / "artifacts"
DEFAULT_STEM = "sfincs_jax_w7x_high_nu_performance"
DEFAULT_RESIDUAL_GATE = 1.0e-6


DEFAULT_RECORDS: list[dict[str, Any]] = [
    {
        "id": "bounded_30k_krylov",
        "label": "30k cap\nKrylov rescue",
        "status": "failed_residual_gate",
        "elapsed_s": 406.9,
        "sparse_factorizations": 0,
        "max_relative_residual": 9.747544e-1,
        "relative_residuals": [7.676920e-1, 8.964357e-1, 9.747544e-1],
        "rhs_elapsed_s": [135.6, 135.6, 135.7],
        "max_rss_mb": 1300.0,
        "note": "bounded 30000-active-DOF cap finished faster but failed the residual gate",
    },
    {
        "id": "sparse_lu_no_reuse",
        "label": "Sparse LU\nno factor reuse",
        "status": "passed",
        "elapsed_s": 2028.0,
        "sparse_factorizations": 3,
        "max_relative_residual": 6.882435e-7,
        "relative_residuals": [6.882435e-7, 7.529734e-9, 7.348069e-9],
        "rhs_elapsed_s": [672.296, 675.450, 677.123],
        "max_rss_mb": 19916.8,
        "note": "residual-clean route before explicit sparse-helper factor reuse",
    },
    {
        "id": "sparse_lu_factor_reuse",
        "label": "Sparse LU\nfactor reuse",
        "status": "passed",
        "elapsed_s": 582.35,
        "sparse_factorizations": 1,
        "max_relative_residual": 6.882435e-7,
        "relative_residuals": [6.882435e-7, 7.529734e-9, 7.348069e-9],
        "rhs_elapsed_s": [573.997, 2.469, 2.378],
        "max_rss_mb": 15319.671875,
        "note": "residual-clean route with block-basis assembly and within-solve factor reuse",
    },
]


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
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9.0,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_w7x_high_nu_performance",
        description="Generate the W7-X high-nu preconditioning/performance figure.",
    )
    parser.add_argument(
        "--records-json",
        type=Path,
        default=None,
        help="Optional JSON file with a list of measured route records.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / f"{DEFAULT_STEM}_summary.json",
    )
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--residual-gate", type=float, default=DEFAULT_RESIDUAL_GATE)
    return parser


def load_records(path: Path | None) -> list[dict[str, Any]]:
    """Load measured records, or return the checked-in default measurements."""

    if path is None:
        return [dict(record) for record in DEFAULT_RECORDS]
    payload = json.loads(Path(path).read_text())
    records = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a list of W7-X performance records")
    return [dict(record) for record in records]


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if np.isfinite(result) else float(default)


def _max_relative_residual(record: dict[str, Any]) -> float:
    explicit = _finite_float(record.get("max_relative_residual"))
    if np.isfinite(explicit):
        return explicit
    values = [_finite_float(value) for value in record.get("relative_residuals", [])]
    values = [value for value in values if np.isfinite(value)]
    return float(max(values)) if values else float("nan")


def build_w7x_high_nu_performance_summary(
    records: list[dict[str, Any]],
    *,
    residual_gate: float = DEFAULT_RESIDUAL_GATE,
    records_source: str = "in_memory_records",
    summary_artifact_checked_in: bool = False,
) -> dict[str, Any]:
    """Return a summary payload with residual, runtime, and factorization gates."""

    normalized: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        row = dict(record)
        row["elapsed_s"] = _finite_float(row.get("elapsed_s"))
        row["elapsed_min"] = float(row["elapsed_s"]) / 60.0
        row["max_relative_residual"] = _max_relative_residual(row)
        row["residual_gate_passed"] = bool(row["max_relative_residual"] <= float(residual_gate))
        row["sparse_factorizations"] = int(row.get("sparse_factorizations", 0) or 0)
        row["max_rss_mb"] = _finite_float(row.get("max_rss_mb"))
        row["rhs_elapsed_s"] = [_finite_float(value, default=0.0) for value in row.get("rhs_elapsed_s", [])]
        normalized.append(row)
        by_id[str(row.get("id"))] = row

    no_reuse = by_id.get("sparse_lu_no_reuse")
    factor_reuse = by_id.get("sparse_lu_factor_reuse")
    failed_route = by_id.get("bounded_30k_krylov")
    speedup = float("nan")
    saved_s = float("nan")
    if no_reuse is not None and factor_reuse is not None:
        no_reuse_elapsed = _finite_float(no_reuse.get("elapsed_s"))
        reuse_elapsed = _finite_float(factor_reuse.get("elapsed_s"))
        if no_reuse_elapsed > 0.0 and reuse_elapsed > 0.0:
            speedup = float(no_reuse_elapsed / reuse_elapsed)
            saved_s = float(no_reuse_elapsed - reuse_elapsed)

    declared_passed_routes = [
        row
        for row in normalized
        if str(row.get("status", "")).lower() in {"passed", "success", "residual_clean"}
    ]
    declared_passed_routes_residual_clean = bool(
        declared_passed_routes and all(bool(row.get("residual_gate_passed")) for row in declared_passed_routes)
    )
    failed_route_rejected = bool(
        failed_route is not None and not bool(failed_route.get("residual_gate_passed"))
    )
    factor_reuse_residual_clean = bool(
        factor_reuse is not None and bool(factor_reuse.get("residual_gate_passed"))
    )
    factor_reuse_fewer_factorizations = bool(
        no_reuse is not None
        and factor_reuse is not None
        and int(factor_reuse.get("sparse_factorizations", 0))
        < int(no_reuse.get("sparse_factorizations", 0))
    )
    factor_reuse_residuals_match_no_reuse = bool(
        no_reuse is not None
        and factor_reuse is not None
        and _residual_series_match(no_reuse, factor_reuse)
    )
    finite_factor_reuse_measurement = bool(
        factor_reuse is not None
        and np.isfinite(_finite_float(factor_reuse.get("elapsed_s")))
        and _finite_float(factor_reuse.get("elapsed_s")) > 0.0
        and np.isfinite(_finite_float(factor_reuse.get("max_rss_mb")))
        and _finite_float(factor_reuse.get("max_rss_mb")) > 0.0
    )
    single_point_performance_claim_supported = bool(
        factor_reuse_residual_clean
        and factor_reuse_fewer_factorizations
        and np.isfinite(speedup)
        and speedup > 1.0
        and failed_route_rejected
        and declared_passed_routes_residual_clean
        and finite_factor_reuse_measurement
    )
    checked_in_converged_artifact = bool(
        single_point_performance_claim_supported
        and factor_reuse_residuals_match_no_reuse
        and str(records_source) == "checked_in_default_records"
        and bool(summary_artifact_checked_in)
    )
    payload: dict[str, Any] = {
        "metadata": {
            "kind": "w7x_high_nu_preconditioning_performance",
            "schema_version": 1,
            "case": "W7-X FP high-nu first point",
            "nuprime": 17.78332923601508,
            "residual_gate": float(residual_gate),
            "records_source": str(records_source),
            "summary_artifact_checked_in": bool(summary_artifact_checked_in),
            "source_script": "examples/publication_figures/generate_w7x_high_nu_performance.py",
            "validation_state": (
                "checked_converged_single_point_performance"
                if checked_in_converged_artifact
                else "performance_claim_deferred_or_external_records"
            ),
            "publication_figure": {
                "claim_status": (
                    "checked_in_converged_artifact"
                    if checked_in_converged_artifact
                    else "proxy_or_deferred"
                ),
                "artifact_class": (
                    "checked_in_w7x_high_nu_single_point_performance"
                    if checked_in_converged_artifact
                    else "external_or_incomplete_w7x_high_nu_performance"
                ),
                "checked_in_converged_artifact": bool(checked_in_converged_artifact),
                "ready_for_physics_validation_claim": False,
                "manuscript_label": "single-point W7-X high-nu preconditioner performance, not physics validation",
            },
            "notes": [
                "This artifact supports a single W7-X high-nu preconditioner performance claim.",
                "It is not a closed W7-X physics-validation or Simakov-Helander analytic-limit claim.",
                "The rejected bounded Krylov route remains plotted only because it fails the residual gate.",
            ],
        },
        "records": normalized,
        "gates": {
            "factor_reuse_present": factor_reuse is not None,
            "factor_reuse_residual_clean": bool(factor_reuse_residual_clean),
            "factor_reuse_fewer_factorizations": bool(factor_reuse_fewer_factorizations),
            "factor_reuse_residuals_match_no_reuse": bool(factor_reuse_residuals_match_no_reuse),
            "factor_reuse_speedup_vs_no_reuse": speedup,
            "factor_reuse_wall_time_saved_s": saved_s,
            "failed_route_rejected": bool(failed_route_rejected),
            "declared_passed_routes_residual_clean": bool(declared_passed_routes_residual_clean),
            "finite_factor_reuse_measurement": bool(finite_factor_reuse_measurement),
            "single_point_performance_claim_supported": bool(single_point_performance_claim_supported),
            "checked_in_converged_artifact": bool(checked_in_converged_artifact),
            "ready_for_physics_validation_claim": False,
        },
    }
    return payload


def write_summary_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _record_values(payload: dict[str, Any], key: str) -> list[float]:
    return [_finite_float(row.get(key), default=0.0) for row in payload["records"]]


def _residual_series_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = [_finite_float(value) for value in left.get("relative_residuals", [])]
    right_values = [_finite_float(value) for value in right.get("relative_residuals", [])]
    left_values = [value for value in left_values if np.isfinite(value)]
    right_values = [value for value in right_values if np.isfinite(value)]
    if not left_values or len(left_values) != len(right_values):
        return False
    return bool(np.allclose(left_values, right_values, rtol=0.0, atol=0.0))


def _plot_runtime_panel(ax, payload: dict[str, Any]) -> None:
    records = payload["records"]
    labels = [str(row["label"]) for row in records]
    minutes = np.asarray(_record_values(payload, "elapsed_min"), dtype=np.float64)
    colors = ["#9ca3af", "#0f4c81", "#1b9e77"]
    x = np.arange(len(records))
    bars = ax.bar(x, minutes, color=colors[: len(records)], edgecolor="black", linewidth=0.4)
    for bar, row in zip(bars, records):
        label = f"{float(row['elapsed_min']):.1f} min"
        if str(row.get("id")) == "sparse_lu_factor_reuse":
            speedup = payload["gates"]["factor_reuse_speedup_vs_no_reuse"]
            if np.isfinite(float(speedup)) and float(speedup) > 1.0:
                label += f"\n{float(speedup):.2f}x faster"
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() * 1.02, label, ha="center", va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylabel("wall time [min]")
    ax.set_title("A. Full W7-X high-nu point runtime")
    ax.set_ylim(0.0, max(minutes) * 1.32)


def _plot_residual_panel(ax, payload: dict[str, Any]) -> None:
    records = payload["records"]
    labels = [str(row["label"]) for row in records]
    residuals = np.asarray(_record_values(payload, "max_relative_residual"), dtype=np.float64)
    x = np.arange(len(records))
    colors = ["#9ca3af", "#0f4c81", "#1b9e77"]
    ax.bar(x, residuals, color=colors[: len(records)], edgecolor="black", linewidth=0.4)
    gate = float(payload["metadata"]["residual_gate"])
    ax.axhline(gate, color="#b91c1c", linestyle="--", linewidth=1.4, label=f"gate {gate:.0e}")
    for idx, value in enumerate(residuals):
        ax.text(idx, value * 1.8, f"{value:.1e}", ha="center", va="bottom", fontsize=8.2)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("max relative residual")
    ax.set_title("B. Residual gate")
    ax.legend(loc="upper right")
    ax.grid(True, which="both", alpha=0.24)


def _plot_factorization_panel(ax, payload: dict[str, Any]) -> None:
    records = [row for row in payload["records"] if str(row.get("id", "")).startswith("sparse_lu")]
    labels = [str(row["label"]) for row in records]
    factors = np.asarray([int(row["sparse_factorizations"]) for row in records], dtype=np.float64)
    x = np.arange(len(records))
    ax.bar(x, factors, color=["#0f4c81", "#1b9e77"][: len(records)], edgecolor="black", linewidth=0.4)
    for idx, value in enumerate(factors):
        ax.text(idx, value + 0.05, f"{int(value)}", ha="center", va="bottom")
    ax.set_xticks(x, labels)
    ax.set_ylabel("host sparse factorizations")
    ax.set_title("C. Reuse removes repeated setup")
    ax.set_ylim(0.0, max(3.4, float(np.max(factors)) * 1.28))


def _plot_memory_panel(ax, payload: dict[str, Any]) -> None:
    records = payload["records"]
    labels = [str(row["label"]) for row in records]
    rss_gb = np.asarray(_record_values(payload, "max_rss_mb"), dtype=np.float64) / 1024.0
    colors = ["#9ca3af", "#0f4c81", "#1b9e77"]
    x = np.arange(len(records))
    ax.bar(x, rss_gb, color=colors[: len(records)], edgecolor="black", linewidth=0.4)
    for idx, value in enumerate(rss_gb):
        if np.isfinite(value) and value > 0.0:
            ax.text(idx, value * 1.03, f"{value:.1f} GB", ha="center", va="bottom", fontsize=8.2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("peak RSS [GB]")
    ax.set_title("D. Measured host memory")
    ax.set_ylim(0.0, max(float(np.nanmax(rss_gb)) * 1.32, 1.0))


def plot_w7x_high_nu_performance(*, payload: dict[str, Any], out_dir: Path, stem: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True)
    _plot_runtime_panel(axes[0, 0], payload)
    _plot_residual_panel(axes[0, 1], payload)
    _plot_factorization_panel(axes[1, 0], payload)
    _plot_memory_panel(axes[1, 1], payload)
    fig.suptitle("W7-X FP high-nu preconditioning: residual-clean factor reuse", fontsize=12.5)
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    _setup_mpl()
    args = _build_parser().parse_args(argv)
    summary_path = Path(args.summary_json)
    default_summary_path = DEFAULT_ARTIFACT_DIR / f"{DEFAULT_STEM}_summary.json"
    payload = build_w7x_high_nu_performance_summary(
        load_records(args.records_json),
        residual_gate=float(args.residual_gate),
        records_source=(
            "checked_in_default_records" if args.records_json is None else "external_records_json"
        ),
        summary_artifact_checked_in=bool(
            args.records_json is None and summary_path.resolve() == default_summary_path.resolve()
        ),
    )
    write_summary_json(summary_path, payload)
    plot_w7x_high_nu_performance(payload=payload, out_dir=Path(args.out_dir), stem=str(args.stem))
    print(f"Wrote W7-X high-nu performance summary to {Path(args.summary_json)}")
    print(f"Wrote W7-X high-nu performance figure to {Path(args.out_dir) / (str(args.stem) + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
