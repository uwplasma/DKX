#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
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
    raise SystemExit("This example requires matplotlib. Install sfincs_jax first.") from exc

from sfincs_jax.validation_artifacts import (
    DEFAULT_PUBLICATION_ARTIFACTS,
    TRANSPORT_ELEMENTS,
    build_simakov_helander_limit_audit_summary,
    collisionality_power_law_slope,
    load_collisionality_records,
    transport_element_abs_series,
)


DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "examples" / "publication_figures" / "artifacts"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "docs" / "_static" / "figures" / "paper"
DEFAULT_SUMMARY_JSON = DEFAULT_ARTIFACT_DIR / "sfincs_jax_simakov_helander_limit_audit_summary.json"
DEFAULT_STEM = "sfincs_jax_simakov_helander_limit_audit"
DEFAULT_LHD_GEOMETRY_OUTPUT = _REPO_ROOT / "examples" / "publication_figures" / "output" / "lhd_co0" / "nu_n_2.668" / "sfincsOutput.h5"
DEFAULT_W7X_GEOMETRY_OUTPUT = _REPO_ROOT / "examples" / "publication_figures" / "output" / "w7x_co0" / "nu_n_1.727" / "sfincsOutput.h5"


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
        prog="generate_simakov_helander_limit_audit",
        description="Generate the bounded Simakov-Helander high-collisionality normalization audit.",
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--n-fit", type=int, default=3)
    parser.add_argument("--min-nuprime-for-full-limit", type=float, default=50.0)
    parser.add_argument("--lhd-geometry-output", type=Path, default=DEFAULT_LHD_GEOMETRY_OUTPUT)
    parser.add_argument("--w7x-geometry-output", type=Path, default=DEFAULT_W7X_GEOMETRY_OUTPUT)
    return parser


def write_audit_summary(
    *,
    artifact_dir: Path,
    summary_json: Path,
    lhd_geometry_output: Path,
    w7x_geometry_output: Path,
    n_fit: int,
    min_nuprime_for_full_limit: float,
    fallback_summary_json: Path = DEFAULT_SUMMARY_JSON,
) -> dict[str, object]:
    precomputed = _load_precomputed_geometry_audits(
        fallback_summary_json=Path(fallback_summary_json),
        lhd_geometry_output=Path(lhd_geometry_output),
        w7x_geometry_output=Path(w7x_geometry_output),
    )
    payload = build_simakov_helander_limit_audit_summary(
        artifact_dir=Path(artifact_dir),
        geometry_outputs={"lhd": Path(lhd_geometry_output), "w7x": Path(w7x_geometry_output)},
        precomputed_geometry_audits=precomputed,
        n_fit=int(n_fit),
        min_nuprime_for_full_limit=float(min_nuprime_for_full_limit),
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _load_precomputed_geometry_audits(
    *,
    fallback_summary_json: Path,
    lhd_geometry_output: Path,
    w7x_geometry_output: Path,
) -> dict[str, dict[str, object]]:
    """Load checked-in geometry audits when bulky HDF5 outputs are unavailable."""

    if Path(lhd_geometry_output).exists() and Path(w7x_geometry_output).exists():
        return {}
    if not Path(fallback_summary_json).exists():
        return {}
    payload = json.loads(Path(fallback_summary_json).read_text())
    if payload.get("metadata", {}).get("kind") != "simakov_helander_limit_audit":
        return {}
    audits: dict[str, dict[str, object]] = {}
    for case in ("lhd", "w7x"):
        audit = payload.get("cases", {}).get(case, {}).get("appendix_b_geometry_audit")
        if isinstance(audit, dict):
            audits[case] = audit
    return audits


def _plot_inverse_tail_panel(ax, *, artifact_dir: Path) -> None:
    colors = {"LHD": "#0f4c81", "W7-X": "#d95f02"}
    for case_label, artifact_name in (
        ("LHD", DEFAULT_PUBLICATION_ARTIFACTS["lhd_collisionality"]),
        ("W7-X", DEFAULT_PUBLICATION_ARTIFACTS["w7x_collisionality"]),
    ):
        records = load_collisionality_records(artifact_dir / artifact_name)
        nuprime, values = transport_element_abs_series(records, label="Fokker-Planck", element=TRANSPORT_ELEMENTS["L11"])
        ax.loglog(nuprime, values, marker="o", color=colors[case_label], label=f"{case_label} FP L11")
        reference = values[-1] * (nuprime / nuprime[-1]) ** -1.0
        ax.loglog(nuprime, reference, linestyle="--", color=colors[case_label], alpha=0.72, label=f"{case_label} nu^-1")
    ax.set_title("FP L11 tail vs inverse-nu reference")
    ax.set_xlabel(r"normalized collisionality $\nu'$")
    ax.set_ylabel(r"$|L_{11}|$")
    ax.grid(True, which="both", alpha=0.24)
    ax.legend(loc="best")


def _plot_slope_panel(ax, *, payload: dict[str, object]) -> None:
    x_positions: list[float] = []
    heights: list[float] = []
    colors: list[str] = []
    labels: list[str] = []
    idx = 0
    for case_name, case_payload in payload["cases"].items():  # type: ignore[union-attr]
        trend = case_payload["trend"]  # type: ignore[index]
        for element_name in ("L11", "L12"):
            x_positions.append(float(idx))
            heights.append(float(trend["slopes"]["Fokker-Planck"][element_name]))  # type: ignore[index]
            colors.append("#0f4c81" if case_name == "lhd" else "#d95f02")
            labels.append(f"{case_name.upper()} {element_name}")
            idx += 1
    ax.axhline(-1.0, color="black", linewidth=1.2, linestyle="--", label="target -1")
    ax.axhspan(-1.35, -0.65, color="black", alpha=0.08, label="audit tolerance")
    ax.bar(x_positions, heights, color=colors, alpha=0.88)
    ax.set_xticks(x_positions, labels, rotation=25, ha="right")
    ax.set_ylabel("tail slope")
    ax.set_title("High-nu slope gate")
    ax.legend(loc="lower left")


def _plot_fit_window_panel(ax, *, artifact_dir: Path) -> None:
    colors = {
        ("LHD", "L11"): "#0f4c81",
        ("LHD", "L12"): "#78a6c8",
        ("W7-X", "L11"): "#d95f02",
        ("W7-X", "L12"): "#f4a261",
    }
    for case_label, artifact_name in (
        ("LHD", DEFAULT_PUBLICATION_ARTIFACTS["lhd_collisionality"]),
        ("W7-X", DEFAULT_PUBLICATION_ARTIFACTS["w7x_collisionality"]),
    ):
        records = load_collisionality_records(artifact_dir / artifact_name)
        for element_name in ("L11", "L12"):
            n_fit_values: list[int] = []
            slopes: list[float] = []
            for n_fit in (2, 3, 4, 5):
                try:
                    slopes.append(
                        collisionality_power_law_slope(
                            records,
                            label="Fokker-Planck",
                            element=TRANSPORT_ELEMENTS[element_name],
                            n_fit=n_fit,
                        )
                    )
                except ValueError:
                    continue
                n_fit_values.append(n_fit)
            ax.plot(
                n_fit_values,
                slopes,
                marker="o",
                color=colors[(case_label, element_name)],
                label=f"{case_label} {element_name}",
            )
    ax.axhline(-1.0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("tail fit points")
    ax.set_ylabel("FP slope")
    ax.set_title("Fit-window sensitivity")
    ax.legend(loc="best", ncols=2)


def _plot_readiness_panel(ax, *, payload: dict[str, object]) -> None:
    ax.axis("off")
    rows = []
    for case_name, case_payload in payload["cases"].items():  # type: ignore[union-attr]
        gates = case_payload["gates"]  # type: ignore[index]
        geom = case_payload["appendix_b_geometry_audit"]  # type: ignore[index]
        fsab_err = float("nan")
        if geom is not None:
            fsab_err = float(geom["geometry_scalars"]["FSABHat2_relative_error"])  # type: ignore[index]
        rows.append(
            [
                case_name.upper(),
                f"{float(case_payload['max_nuprime']):.3g}",  # type: ignore[index]
                "yes" if gates["fp_l11_l12_target_inverse_slope"] else "no",  # type: ignore[index]
                "yes" if gates["scan_extends_to_required_high_nu"] else "no",  # type: ignore[index]
                f"{fsab_err:.1e}",
            ]
        )
    table = ax.table(
        cellText=rows,
        colLabels=["case", "max nu'", "FP slope", "nu' >= gate", "FSAB2 err"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.4)
    gate = payload["configuration"]["min_nuprime_for_full_limit"]  # type: ignore[index]
    ax.set_title(f"Appendix-B readiness audit (full gate: nu' >= {float(gate):.0f})")
    ax.text(
        0.02,
        0.04,
        "Full Simakov-Helander reproduction remains closed until wider high-nu scans are pinned.",
        transform=ax.transAxes,
        fontsize=8.5,
    )


def plot_audit(*, payload: dict[str, object], artifact_dir: Path, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8), constrained_layout=True)
    _plot_inverse_tail_panel(axes[0, 0], artifact_dir=Path(artifact_dir))
    _plot_slope_panel(axes[0, 1], payload=payload)
    _plot_fit_window_panel(axes[1, 0], artifact_dir=Path(artifact_dir))
    _plot_readiness_panel(axes[1, 1], payload=payload)
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    _setup_mpl()
    args = _build_parser().parse_args(argv)
    payload = write_audit_summary(
        artifact_dir=Path(args.artifact_dir),
        summary_json=Path(args.summary_json),
        lhd_geometry_output=Path(args.lhd_geometry_output),
        w7x_geometry_output=Path(args.w7x_geometry_output),
        n_fit=int(args.n_fit),
        min_nuprime_for_full_limit=float(args.min_nuprime_for_full_limit),
        fallback_summary_json=DEFAULT_SUMMARY_JSON,
    )
    plot_audit(payload=payload, artifact_dir=Path(args.artifact_dir), out_dir=Path(args.out_dir), stem=str(args.stem))
    print(f"Wrote Simakov-Helander audit summary to {Path(args.summary_json)}")
    print(f"Wrote Simakov-Helander audit figures to {Path(args.out_dir) / (str(args.stem) + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
