#!/usr/bin/env python
"""Plot finite-beta VMEX/DKX resolution sensitivity from cached runs.

The finite-beta transport example can be expensive at high velocity-space
resolution. This helper reads the cached ``sfincsOutput.h5`` files produced by
``finite_beta_vmec_to_sfincs.py`` and summarizes which numerical parameters move
the ambipolar root and bootstrap current most strongly.

Run from the repository root after generating the finite-beta example outputs:

    python examples/vmex_finite_beta/plot_convergence_scan.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = EXAMPLE_DIR / "output"
DEFAULT_STEM = "finite_beta_vmex_sfincs_convergence_scan"


def _load_pipeline_module() -> ModuleType:
    script = EXAMPLE_DIR / "finite_beta_vmec_to_sfincs.py"
    spec = importlib.util.spec_from_file_location("finite_beta_vmec_to_sfincs", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _er_from_dir_name(name: str) -> float:
    if not name.startswith("Er"):
        raise ValueError(name)
    value = name[2:]
    if value == "":
        raise ValueError(name)
    return float(value.replace("m", "-"))


def _records_from_scan(mod: ModuleType, scan_root: Path, r_n: float) -> list[object]:
    r_dir = scan_root / mod._format_r_dir(r_n)
    if not r_dir.exists():
        return []
    records = []
    for output_h5 in sorted(r_dir.glob("Er*/sfincsOutput.h5")):
        try:
            er = _er_from_dir_name(output_h5.parent.name)
        except ValueError:
            continue
        records.append(mod._record_from_output(r_n=float(r_n), er=er, output_h5=output_h5))
    return sorted(records, key=lambda record: record.er)


def _filter_by_er_spacing(records: Iterable[object], spacing: float | None) -> list[object]:
    records = list(records)
    if spacing is None:
        return sorted(records, key=lambda record: record.er)
    allowed = []
    for record in records:
        scaled = float(record.er) / float(spacing)
        if abs(scaled - round(scaled)) <= 1.0e-8:
            allowed.append(record)
    return sorted(allowed, key=lambda record: record.er)


def _single_surface_profile(mod: ModuleType, records: list[object], r_n: float, preferred_er: float = 0.0):
    if not records:
        return None
    return mod.build_radial_profile(scans_by_radius={float(r_n): (records, Path("."))}, preferred_er=preferred_er)[0]


def _profile_series(mod: ModuleType, scan_root: Path, r_values: list[float], spacing: float | None = None):
    profiles = []
    for r_n in r_values:
        records = _filter_by_er_spacing(_records_from_scan(mod, scan_root, r_n), spacing)
        profile = _single_surface_profile(mod, records, r_n)
        if profile is not None and profile.selected_ambipolar_er is not None:
            profiles.append(profile)
    return profiles


def _max_abs_delta(a: list[object], b: list[object], attr: str) -> float | None:
    a_by_r = {round(float(item.r_n), 12): item for item in a}
    b_by_r = {round(float(item.r_n), 12): item for item in b}
    deltas = []
    for key in sorted(set(a_by_r).intersection(b_by_r)):
        va = getattr(a_by_r[key], attr)
        vb = getattr(b_by_r[key], attr)
        if va is None or vb is None:
            continue
        deltas.append(abs(float(va) - float(vb)))
    return float(max(deltas)) if deltas else None


def _profile_dicts(profiles: list[object]) -> list[dict[str, object]]:
    return [asdict(profile) for profile in profiles]


def build_convergence_payload(out_dir: Path) -> dict[str, object]:
    mod = _load_pipeline_module()
    r_values = [0.15, 0.30, 0.50, 0.70, 0.85]
    sensitive_r = [0.50, 0.85]

    kinetic_configs = [
        ("5/4/4", out_dir / "sfincs_radial_er_scan_Nt7_Nz7_Nxi5_NL4_Nx4"),
        ("6/5/5", out_dir / "sfincs_radial_er_scan_Nt7_Nz7_Nxi6_NL5_Nx5"),
        ("7/6/6", out_dir / "sfincs_radial_er_scan_Nt7_Nz7_Nxi7_NL6_Nx6"),
        ("8/6/6", out_dir / "sfincs_radial_er_scan_Nt7_Nz7_Nxi8_NL6_Nx6"),
    ]
    kinetic_profiles: dict[str, list[dict[str, object]]] = {}
    for label, scan_root in kinetic_configs:
        profiles = _profile_series(mod, scan_root, sensitive_r)
        if profiles:
            kinetic_profiles[label] = _profile_dicts(profiles)

    angular_profiles: dict[str, list[dict[str, object]]] = {}
    for label, scan_root in [
        ("7x7,6/5/5", out_dir / "sfincs_radial_er_scan_Nt7_Nz7_Nxi6_NL5_Nx5"),
        ("9x9,6/5/5", out_dir / "sfincs_radial_er_scan_convergence_Nt9_Nz9_Nxi6_NL5_Nx5"),
    ]:
        profiles = _profile_series(mod, scan_root, [0.15, 0.50, 0.85])
        if profiles:
            angular_profiles[label] = _profile_dicts(profiles)

    high_base_profiles = _profile_series(mod, out_dir / "sfincs_radial_er_scan_Nt7_Nz7_Nxi8_NL6_Nx6", r_values)
    high_refined_profiles = _profile_series(
        mod,
        out_dir / "sfincs_radial_er_scan_convergence_Nt7_Nz7_Nxi8_NL6_Nx6",
        r_values,
    )

    parameter_probes: dict[str, dict[str, object]] = {}
    reference_probe_path = EXAMPLE_DIR / "reference" / "finite_beta_parameter_probes.json"
    if reference_probe_path.exists():
        try:
            reference_probe = json.loads(reference_probe_path.read_text(encoding="utf-8"))
            for label, row in reference_probe.get("profiles", {}).items():
                parameter_probes[str(label)] = {
                    "r_n": float(reference_probe.get("r_n", 0.5)),
                    "psi_n": float(reference_probe.get("r_n", 0.5)) ** 2,
                    "roots_er": row.get("roots_er", []),
                    "bootstrap_current_at_roots": row.get("bootstrap_current_at_roots", []),
                    "selected_ambipolar_er": row.get("selected_ambipolar_er"),
                    "selected_bootstrap_current": row.get("selected_bootstrap_current"),
                    "scan_dir": str(reference_probe_path),
                }
        except Exception:
            parameter_probes = {}
    baseline_profiles = _profile_series(mod, out_dir / "sfincs_radial_er_scan_Nt7_Nz7_Nxi7_NL6_Nx6", [0.50])
    if baseline_profiles:
        parameter_probes["7/6/6"] = asdict(baseline_profiles[0])
    for label, summary_name in [
        ("8/6/6", "param_nxi8_nl6_nx6_summary.json"),
        ("7/7/6", "param_nxi7_nl7_nx6_summary.json"),
        ("7/6/7", "param_nxi7_nl6_nx7_summary.json"),
    ]:
        summary_path = out_dir / summary_name
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = summary["dkx"]["radial_profile"]
        except Exception:
            continue
        if rows:
            parameter_probes[label] = rows[0]

    deltas = {}
    if "7/6/6" in kinetic_profiles and "8/6/6" in kinetic_profiles:
        p0 = [mod.SurfaceProfileRecord(**row) for row in kinetic_profiles["7/6/6"]]
        p1 = [mod.SurfaceProfileRecord(**row) for row in kinetic_profiles["8/6/6"]]
        deltas["kinetic 7/6/6 -> 8/6/6"] = {
            "max_abs_er": _max_abs_delta(p0, p1, "selected_ambipolar_er"),
            "max_abs_bootstrap": _max_abs_delta(p0, p1, "selected_bootstrap_current"),
        }
    if "7x7,6/5/5" in angular_profiles and "9x9,6/5/5" in angular_profiles:
        p0 = [mod.SurfaceProfileRecord(**row) for row in angular_profiles["7x7,6/5/5"]]
        p1 = [mod.SurfaceProfileRecord(**row) for row in angular_profiles["9x9,6/5/5"]]
        deltas["angular 7x7 -> 9x9 at 6/5/5"] = {
            "max_abs_er": _max_abs_delta(p0, p1, "selected_ambipolar_er"),
            "max_abs_bootstrap": _max_abs_delta(p0, p1, "selected_bootstrap_current"),
        }
    if high_base_profiles and high_refined_profiles:
        deltas["Er bracket 1.25 -> 0.625 at 8/6/6"] = {
            "max_abs_er": _max_abs_delta(high_base_profiles, high_refined_profiles, "selected_ambipolar_er"),
            "max_abs_bootstrap": _max_abs_delta(
                high_base_profiles,
                high_refined_profiles,
                "selected_bootstrap_current",
            ),
        }

    if "7/6/6" in parameter_probes:
        base = parameter_probes["7/6/6"]
        base_er = _row_value(base, "selected_ambipolar_er")
        base_j = _row_value(base, "selected_bootstrap_current")
        for label in ["8/6/6", "7/7/6", "7/6/7"]:
            row = parameter_probes.get(label)
            if row is None:
                continue
            deltas[f"single-parameter {label} at rN=0.5"] = {
                "max_abs_er": abs(_row_value(row, "selected_ambipolar_er") - base_er),
                "max_abs_bootstrap": abs(_row_value(row, "selected_bootstrap_current") - base_j),
            }

    high_summary = None
    if high_base_profiles and high_refined_profiles:
        high_summary = {
            "max_abs_er": _max_abs_delta(high_base_profiles, high_refined_profiles, "selected_ambipolar_er"),
            "max_abs_bootstrap": _max_abs_delta(
                high_base_profiles,
                high_refined_profiles,
                "selected_bootstrap_current",
            ),
        }

    return {
        "kinetic_profiles": kinetic_profiles,
        "angular_profiles": angular_profiles,
        "high_base_profile": _profile_dicts(high_base_profiles),
        "high_refined_profile": _profile_dicts(high_refined_profiles),
        "high_bracket_summary": high_summary,
        "parameter_probes": parameter_probes,
        "deltas": deltas,
        "note": (
            "The highest full-profile grid generated for this documentation run is "
            "Ntheta=7, Nzeta=7, Nxi=8, NL=6, Nx=6, with a tighter same-grid Er bracket "
            "refinement from width 1.25 to 0.625. Nxi=9, NL=6, Nx=6 and Nxi=8, NL=6, "
            "Nx=7 were probed on an RTX A4000 but were too expensive for the bounded "
            "documentation campaign."
        ),
    }


def _row_value(row: dict[str, object], key: str) -> float:
    value = row[key]
    if value is None:
        return float("nan")
    return float(value)


def plot_convergence_payload(payload: dict[str, object], out_dir: Path, stem: str) -> tuple[Path, Path]:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    colors = {"0.5": "#284b63", "0.85": "#b24c36", "all": "#d97706"}
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.2), constrained_layout=True)

    kinetic = payload["kinetic_profiles"]
    labels = list(kinetic)
    x = np.arange(len(labels), dtype=float)
    for r_n in [0.50, 0.85]:
        values = []
        for label in labels:
            rows = [row for row in kinetic[label] if abs(float(row["r_n"]) - r_n) < 1.0e-10]
            values.append(_row_value(rows[0], "selected_ambipolar_er") if rows else np.nan)
        axes[0, 0].plot(x, values, "o-", label=rf"$r_N={r_n:.2f}$", color=colors[f"{r_n:g}"])
    axes[0, 0].set_xticks(x, labels)
    axes[0, 0].set_title("Kinetic-space convergence of ambipolar root")
    axes[0, 0].set_xlabel(r"$N_\xi/N_L/N_x$")
    axes[0, 0].set_ylabel(r"selected ambipolar $E_r$")
    axes[0, 0].legend(loc="best")

    for r_n in [0.50, 0.85]:
        values = []
        for label in labels:
            rows = [row for row in kinetic[label] if abs(float(row["r_n"]) - r_n) < 1.0e-10]
            values.append(_row_value(rows[0], "selected_bootstrap_current") if rows else np.nan)
        axes[0, 1].plot(x, values, "s-", label=rf"$r_N={r_n:.2f}$", color=colors[f"{r_n:g}"])
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].set_title("Kinetic-space convergence of bootstrap current")
    axes[0, 1].set_xlabel(r"$N_\xi/N_L/N_x$")
    axes[0, 1].set_ylabel("FSABjHat at ambipolar root")
    axes[0, 1].legend(loc="best")

    high_base = payload["high_base_profile"]
    high_refined = payload["high_refined_profile"]
    for rows, style, label in [
        (high_base, "o-", "width 1.25"),
        (high_refined, "s--", "width 0.625"),
    ]:
        x_high = np.asarray([_row_value(row, "psi_n") for row in rows], dtype=float)
        y_high = np.asarray([_row_value(row, "selected_bootstrap_current") for row in rows], dtype=float)
        axes[1, 0].plot(x_high, y_high, style, label=label)
    high_summary = payload.get("high_bracket_summary") or {}
    max_er = high_summary.get("max_abs_er")
    max_j = high_summary.get("max_abs_bootstrap")
    suffix = ""
    if max_er is not None and max_j is not None:
        suffix = rf"max $|\Delta E_r|={float(max_er):.1e}$, max $|\Delta J|={float(max_j):.1e}$"
    axes[1, 0].set_title(r"Same-grid root-bracket convergence at $8/6/6$")
    axes[1, 0].set_xlabel(r"normalized toroidal flux $\psi_N$")
    axes[1, 0].set_ylabel("FSABjHat at ambipolar root")
    axes[1, 0].legend(loc="best")
    if suffix:
        axes[1, 0].text(0.02, 0.95, suffix, transform=axes[1, 0].transAxes, va="top", fontsize=9.0)

    parameter_probes = payload["parameter_probes"]
    probe_labels = [label for label in ["7/7/6", "7/6/7", "8/6/6"] if label in parameter_probes]
    base = parameter_probes.get("7/6/6", {})
    base_er = _row_value(base, "selected_ambipolar_er") if base else np.nan
    delta_labels = [f"{label}\nvs 7/6/6" for label in probe_labels]
    delta_values = [
        abs(_row_value(parameter_probes[label], "selected_ambipolar_er") - base_er)
        for label in probe_labels
    ]
    bar_x = np.arange(len(delta_labels), dtype=float)
    axes[1, 1].bar(bar_x, delta_values, color=["#2a9d8f", "#6d597a", "#d97706"][: len(delta_labels)])
    axes[1, 1].set_xticks(bar_x, delta_labels)
    axes[1, 1].set_title(r"One-parameter probes at $r_N=0.50$")
    axes[1, 1].set_ylabel(r"$|\Delta E_r|$ from $7/6/6$")
    for idx, value in enumerate(delta_values):
        if value is not None and np.isfinite(float(value)):
            axes[1, 1].text(idx, float(value), f"{float(value):.2g}", ha="center", va="bottom", fontsize=8.5)

    fig.suptitle("Finite-beta VMEX to dkx convergence scan", fontsize=14)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png.resolve(), pdf.resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--stem", default=DEFAULT_STEM)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    payload = build_convergence_payload(out_dir)
    png, pdf = plot_convergence_payload(payload, out_dir, str(args.stem))
    json_path = out_dir / f"{args.stem}_summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Convergence scan PNG: {png}")
    print(f"Convergence scan PDF: {pdf}")
    print(f"Convergence scan JSON: {json_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
