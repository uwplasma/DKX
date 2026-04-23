#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

import h5py
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS = REPO_ROOT / "utils"
EXAMPLES = REPO_ROOT / "examples" / "sfincs_examples"


@dataclass(frozen=True)
class ScanConfig:
    name: str
    base_input: Path
    nuprime_factor: float
    collision_operator: int
    label: str


@dataclass(frozen=True)
class TransportScanPoint:
    label: str
    nuprime: float
    transport_matrix: list[list[float]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_sfincs_paper_figs",
        description="Reproduce low-resolution SFINCS paper figures with sfincs_jax runs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "_static" / "figures" / "paper",
        help="Directory for output figures.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / "examples" / "publication_figures" / "output",
        help="Scratch directory for scan runs.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=None,
        help="Directory for machine-readable scan summaries. Defaults to --work-dir.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use reduced resolution and fewer scan points.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=None,
        help="Per-step timeout in seconds (applied to each scan run).",
    )
    parser.add_argument(
        "--case",
        choices=("lhd", "w7x", "all"),
        default="all",
        help="Which geometry scans to run/plot.",
    )
    parser.add_argument(
        "--collision-operators",
        default="0,1",
        help="Comma-separated collision-operator subset to run/collect (default: 0,1).",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Run scans only (skip plotting).",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot only (reuse existing scan output; do not run scans).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse an existing operator scan directory when it already contains sfincsOutput.h5 files.",
    )
    return parser.parse_args()


def _run(cmd: list[str], *, cwd: Path, timeout_s: float | None, label: str) -> None:
    print(f"[{label}] cwd={cwd}")
    print(f"[{label}] cmd={' '.join(cmd)}")
    log_path = cwd / f"{label}.log"
    print(f"[{label}] log={log_path}")
    sys.stdout.flush()
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.setdefault("MPLCONFIGDIR", str(cwd / ".mplconfig"))
    with log_path.open("w") as log:
        subprocess.run(
            cmd,
            cwd=str(cwd),
            check=True,
            env=env,
            timeout=timeout_s,
            stdout=log,
            stderr=log,
        )


def _strip_ss_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.strip().lower().startswith("!ss"):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def _inject_group(text: str, group: str, lines: list[str]) -> str:
    out: list[str] = []
    inserted = False
    for line in text.splitlines():
        out.append(line)
        if line.strip().lower().startswith(f"&{group.lower()}"):
            out.extend(lines)
            inserted = True
    if not inserted:
        out.append(f"&{group}")
        out.extend(lines)
        out.append("/")
    return "\n".join(out) + "\n"


def _set_group_assignment(text: str, group: str, key: str, value: str) -> str:
    group_pattern = re.compile(
        rf"(^\s*&{re.escape(group)}\b.*?$)(.*?)(^\s*/\s*$)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = group_pattern.search(text)
    if not match:
        raise ValueError(f"Could not find &{group} group in input text")
    group_header, group_body, group_end = match.groups()
    assign_pattern = re.compile(rf"(^\s*{re.escape(key)}\s*=\s*)([^!\n\r]*)(.*)$", re.IGNORECASE | re.MULTILINE)
    if assign_pattern.search(group_body):
        new_body = assign_pattern.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", group_body, count=1)
    else:
        if group_body and not group_body.endswith("\n"):
            group_body += "\n"
        new_body = group_body + f"  {key} = {value}\n"
    return text[: match.start()] + group_header + new_body + group_end + text[match.end() :]


def _write_scan_input(
    *,
    base_input: Path,
    dest: Path,
    nu_n_min: float,
    nu_n_max: float,
    n_points: int,
    collision_operator: int,
    fast: bool,
) -> None:
    text = _strip_ss_lines(base_input.read_text())
    text = text + "\n".join(
        [
            "!ss scanType = 3",
            "!ss scanVariable = nu_n",
            f"!ss scanVariableMin = {nu_n_min:.6e}",
            f"!ss scanVariableMax = {nu_n_max:.6e}",
            f"!ss scanVariableN = {n_points}",
            "!ss scanVariableScale = log",
            "",
        ]
    )
    text = _set_group_assignment(text, "physicsParameters", "collisionOperator", str(int(collision_operator)))
    if fast:
        for key, value in (
            ("Ntheta", "5"),
            ("Nzeta", "5"),
            ("Nxi", "3"),
            ("NL", "3"),
            ("Nx", "3"),
            ("solverTolerance", "1e-4"),
        ):
            text = _set_group_assignment(text, "resolutionParameters", key, value)
    dest.write_text(text)


def _collect_transport_matrix(work_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for sub in sorted(work_dir.iterdir()):
        if not sub.is_dir():
            continue
        h5_path = sub / "sfincsOutput.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            nu_n = float(np.asarray(f["nu_n"][()]))
            g_hat = float(np.asarray(f["GHat"][()]))
            i_hat = float(np.asarray(f["IHat"][()]))
            iota = float(np.asarray(f["iota"][()]))
            b0_over_bbar = float(np.asarray(f["B0OverBBar"][()]))
            nuprime = nu_n * (g_hat + iota * i_hat) / b0_over_bbar
            tm = np.asarray(f["transportMatrix"][()], dtype=float)
        rows.append((nuprime, tm))
    rows.sort(key=lambda x: x[0])
    nu = np.asarray([r[0] for r in rows])
    tm = np.asarray([r[1] for r in rows])
    return nu, tm


def _has_transport_outputs(work_dir: Path) -> bool:
    return any(work_dir.glob("*/sfincsOutput.h5"))


def _parse_collision_operators(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return (0, 1)
    out: list[int] = []
    for part in parts:
        op = int(part)
        if op not in {0, 1}:
            raise ValueError(f"Unsupported collision operator {op}; expected 0 and/or 1.")
        if op not in out:
            out.append(op)
    return tuple(out)


def build_transport_scan_summary_rows(
    datasets: dict[str, tuple[np.ndarray, np.ndarray]]
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for label, (nu, tm) in datasets.items():
        for idx in range(len(nu)):
            payload.append(
                {
                    "label": label,
                    "nuprime": float(nu[idx]),
                    "transport_matrix": np.asarray(tm[idx], dtype=float).tolist(),
                }
            )
    payload.sort(key=lambda row: (str(row["label"]), float(row["nuprime"])))
    return payload


def write_transport_scan_summary_json(
    summary_path: Path,
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    metadata: dict[str, object] | None = None,
) -> None:
    rows = build_transport_scan_summary_rows(datasets)
    payload: dict[str, object] | list[dict[str, object]]
    if metadata is None:
        payload = rows
    else:
        payload = {"metadata": metadata, "rows": rows}
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _summary_metadata(
    *,
    case: str,
    fast: bool,
    n_points: int,
    nuprime_min: float,
    nuprime_max: float,
    work_dir: Path,
    summary_path: Path,
    base_input: Path,
    labels_to_collision_operator: dict[str, int],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "case": case,
        "fast": bool(fast),
        "n_points": int(n_points),
        "nuprime_min": float(nuprime_min),
        "nuprime_max": float(nuprime_max),
        "base_input": str(base_input.relative_to(REPO_ROOT)),
        "source_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "work_dir": str(work_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "labels_to_collision_operator": {
            str(label): int(operator) for label, operator in labels_to_collision_operator.items()
        },
    }


def _fit_high_collisionality(nu: np.ndarray, y: np.ndarray, n_fit: int = 2) -> np.ndarray:
    if nu.size < n_fit + 1:
        return y
    x_fit = np.log(nu[-n_fit:])
    y_fit = np.log(np.abs(y[-n_fit:]) + 1e-30)
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    sign = np.sign(y[-1]) if np.sign(y[-1]) != 0 else 1.0
    return sign * np.exp(slope * np.log(nu) + intercept)


def _plot_matrix_elements(
    *,
    out_path: Path,
    title: str,
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    y_label: str = "transportMatrix element",
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True, sharex=True)
    elements = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for ax, (i, j) in zip(axes.flat, elements, strict=False):
        for label, (nu, tm) in datasets.items():
            ax.plot(nu, tm[:, i, j], marker="o", label=label)
        ax.set_xscale("log")
        ax.set_title(f"L{i+1}{j+1}")
        ax.grid(True, which="both", alpha=0.3)
    axes[1, 0].set_xlabel(r"$\nu'$")
    axes[1, 1].set_xlabel(r"$\nu'$")
    axes[0, 0].set_ylabel(y_label)
    axes[1, 0].set_ylabel(y_label)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_simakov_helander_proxy(
    *,
    out_path: Path,
    title: str,
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    element: tuple[int, int] = (0, 0),
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    i, j = element
    for label, (nu, tm) in datasets.items():
        ax.plot(nu, tm[:, i, j], marker="o", label=label)
        ax.plot(nu, _fit_high_collisionality(nu, tm[:, i, j]), linestyle="--", alpha=0.7)
    ax.set_xscale("log")
    ax.set_title(f"{title} (L{i+1}{j+1})")
    ax.set_xlabel(r"$\nu'$")
    ax.set_ylabel("transportMatrix element")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    out_dir = args.out_dir
    work_dir = args.work_dir
    summary_dir = args.summary_dir if args.summary_dir is not None else work_dir
    fast = bool(args.fast)
    timeout_s = args.timeout_s
    case = args.case
    scan_only = bool(args.scan_only)
    plot_only = bool(args.plot_only)
    skip_existing = bool(args.skip_existing)
    selected_collision_operators = set(_parse_collision_operators(args.collision_operators))

    if scan_only and plot_only:
        raise ValueError("Cannot combine --scan-only and --plot-only.")

    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(work_dir / ".mplconfig"))

    if not plot_only and not skip_existing:
        if case in ("lhd", "all"):
            for collision_operator in selected_collision_operators:
                shutil.rmtree(work_dir / f"lhd_co{collision_operator}", ignore_errors=True)
        if case in ("w7x", "all"):
            for collision_operator in selected_collision_operators:
                shutil.rmtree(work_dir / f"w7x_co{collision_operator}", ignore_errors=True)

    n_points = 4 if fast else 7
    nuprime_min = 0.1
    nuprime_max = 10.0

    lhd = ScanConfig(
        name="lhd",
        base_input=EXAMPLES / "transportMatrix_geometryScheme2" / "input.namelist",
        nuprime_factor=0.2668018,
        collision_operator=0,
        label="Fokker-Planck",
    )
    w7x = ScanConfig(
        name="w7x",
        base_input=EXAMPLES / "transportMatrix_geometryScheme11" / "input.namelist",
        nuprime_factor=0.172714565,
        collision_operator=0,
        label="Fokker-Planck",
    )

    fig1_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    if case in ("lhd", "all"):
        scan_models = [
            (lhd, 0, "Fokker-Planck"),
            (lhd, 1, "PAS"),
        ]
        for cfg, collision_operator, label in scan_models:
            if collision_operator not in selected_collision_operators:
                continue
            nu_n_min = nuprime_min * cfg.nuprime_factor
            nu_n_max = nuprime_max * cfg.nuprime_factor
            case_dir = work_dir / f"{cfg.name}_co{collision_operator}"
            case_dir.mkdir(parents=True, exist_ok=True)
            if not plot_only:
                should_run = not (skip_existing and _has_transport_outputs(case_dir))
                if should_run:
                    _write_scan_input(
                        base_input=cfg.base_input,
                        dest=case_dir / "input.namelist",
                        nu_n_min=nu_n_min,
                        nu_n_max=nu_n_max,
                        n_points=n_points,
                        collision_operator=collision_operator,
                        fast=fast,
                    )
                    _run(
                        [sys.executable, str(UTILS / "sfincsScan"), "--yes", "--input", "input.namelist"],
                        cwd=case_dir,
                        timeout_s=timeout_s,
                        label=f"scan-{cfg.name}-co{collision_operator}",
                    )
            if _has_transport_outputs(case_dir):
                fig1_data[label] = _collect_transport_matrix(case_dir)

        if fig1_data:
            lhd_summary_path = summary_dir / f"lhd_collisionality{'_fast' if fast else ''}_summary.json"
            write_transport_scan_summary_json(
                lhd_summary_path,
                fig1_data,
                metadata=_summary_metadata(
                    case="lhd",
                    fast=fast,
                    n_points=n_points,
                    nuprime_min=nuprime_min,
                    nuprime_max=nuprime_max,
                    work_dir=work_dir,
                    summary_path=lhd_summary_path,
                    base_input=lhd.base_input,
                    labels_to_collision_operator={label: operator for label, operator in (("Fokker-Planck", 0), ("PAS", 1)) if operator in selected_collision_operators},
                ),
            )

        if not scan_only and fig1_data:
            _plot_matrix_elements(
                out_path=out_dir / "sfincs_jax_fig1_lhd_collisionality.png",
                title="LHD collisionality scan (sfincs_jax)",
                datasets=fig1_data,
            )

    # Figure 2 (W7-X)
    fig2_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if case in ("w7x", "all"):
        for collision_operator, label in [(0, "Fokker-Planck"), (1, "PAS")]:
            if collision_operator not in selected_collision_operators:
                continue
            nu_n_min = nuprime_min * w7x.nuprime_factor
            nu_n_max = nuprime_max * w7x.nuprime_factor
            case_dir = work_dir / f"w7x_co{collision_operator}"
            case_dir.mkdir(parents=True, exist_ok=True)
            if not plot_only:
                should_run = not (skip_existing and _has_transport_outputs(case_dir))
                if should_run:
                    _write_scan_input(
                        base_input=w7x.base_input,
                        dest=case_dir / "input.namelist",
                        nu_n_min=nu_n_min,
                        nu_n_max=nu_n_max,
                        n_points=n_points,
                        collision_operator=collision_operator,
                        fast=fast,
                    )
                    _run(
                        [sys.executable, str(UTILS / "sfincsScan"), "--yes", "--input", "input.namelist"],
                        cwd=case_dir,
                        timeout_s=timeout_s,
                        label=f"scan-w7x-co{collision_operator}",
                    )
            if _has_transport_outputs(case_dir):
                fig2_data[label] = _collect_transport_matrix(case_dir)

        if fig2_data:
            w7x_summary_path = summary_dir / f"w7x_collisionality{'_fast' if fast else ''}_summary.json"
            write_transport_scan_summary_json(
                w7x_summary_path,
                fig2_data,
                metadata=_summary_metadata(
                    case="w7x",
                    fast=fast,
                    n_points=n_points,
                    nuprime_min=nuprime_min,
                    nuprime_max=nuprime_max,
                    work_dir=work_dir,
                    summary_path=w7x_summary_path,
                    base_input=w7x.base_input,
                    labels_to_collision_operator={label: operator for label, operator in (("Fokker-Planck", 0), ("PAS", 1)) if operator in selected_collision_operators},
                ),
            )

        if not scan_only and fig2_data:
            _plot_matrix_elements(
                out_path=out_dir / "sfincs_jax_fig2_w7x_collisionality.png",
                title="W7-X collisionality scan (sfincs_jax)",
                datasets=fig2_data,
            )

    # Figure 3 proxy: high-collisionality fit for FP data
    if not scan_only and fig1_data and fig2_data:
        fig3_data = {
            "LHD (FP)": fig1_data["Fokker-Planck"],
            "W7-X (FP)": fig2_data["Fokker-Planck"],
        }
        _plot_simakov_helander_proxy(
            out_path=out_dir / "sfincs_jax_fig3_simakov_helander.png",
            title="High-collisionality proxy",
            datasets=fig3_data,
            element=(0, 0),
        )


if __name__ == "__main__":
    main()
