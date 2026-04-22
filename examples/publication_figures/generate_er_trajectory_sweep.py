#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("This example requires matplotlib. Install with: pip install matplotlib") from exc

from sfincs_jax.io import write_sfincs_jax_output_h5
from sfincs_jax.namelist import read_sfincs_input


@dataclass(frozen=True)
class TrajectoryModel:
    name: str
    label: str
    include_xdot_term: bool
    include_electric_field_term_in_xidot: bool
    use_dkes_exb_drift: bool
    magnetic_drift_scheme: int


@dataclass(frozen=True)
class SweepRecord:
    model: str
    label: str
    er: float
    er_over_eres: float | None
    particle_flux_vm_psi_hat: float
    heat_flux_vm_psi_hat: float
    fsab_flow: float
    fsab_jhat: float
    output_path: str


TRAJECTORY_MODELS: tuple[TrajectoryModel, ...] = (
    TrajectoryModel(
        name="dkes",
        label="DKES trajectories",
        include_xdot_term=False,
        include_electric_field_term_in_xidot=False,
        use_dkes_exb_drift=True,
        magnetic_drift_scheme=0,
    ),
    TrajectoryModel(
        name="partial",
        label="Partial trajectories",
        include_xdot_term=False,
        include_electric_field_term_in_xidot=False,
        use_dkes_exb_drift=False,
        magnetic_drift_scheme=0,
    ),
    TrajectoryModel(
        name="full",
        label="Full trajectories",
        include_xdot_term=True,
        include_electric_field_term_in_xidot=True,
        use_dkes_exb_drift=False,
        magnetic_drift_scheme=1,
    ),
)


PRESETS = {
    "tokamak_like": _REPO_ROOT / "examples" / "upstream" / "fortran_v3" / "tokamak_1species_FPCollisions_withEr_fullTrajectories" / "input.namelist",
    "stellarator_like": _REPO_ROOT / "examples" / "upstream" / "fortran_v3" / "sfincsPaperFigure3_geometryScheme11_FPCollisions_2Species_fullTrajectories" / "input.namelist",
}


def _setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "lines.linewidth": 2.0,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="generate_er_trajectory_sweep",
        description="Generate a literature-anchored Er trajectory-model sweep scaffold for sfincs_jax.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Base input.namelist. Overrides --preset.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tokamak_like", help="Convenience preset for a base input.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=_REPO_ROOT / "examples" / "publication_figures" / "output" / "er_sweep",
        help="Directory for generated inputs, outputs, and summary JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "docs" / "_static" / "figures" / "paper",
        help="Directory for publication-style figures.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to the summary JSON. Defaults to WORK_DIR/er_trajectory_sweep_summary.json.",
    )
    parser.add_argument(
        "--er-values",
        default="",
        help="Comma-separated raw Er values. If omitted, use a symmetric ladder around the input Er magnitude.",
    )
    parser.add_argument(
        "--er-res",
        type=float,
        default=None,
        help="Optional resonant electric field used to normalize the x-axis and JSON records.",
    )
    parser.add_argument(
        "--species-index",
        type=int,
        default=0,
        help="Species index for particle/heat flux and flow summaries.",
    )
    parser.add_argument("--fast", action="store_true", help="Apply a bounded low-resolution override for quicker exploratory sweeps.")
    parser.add_argument("--plot-only", action="store_true", help="Reuse an existing summary JSON and only regenerate the figure.")
    return parser.parse_args()


def _replace_assignment(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(^\s*{re.escape(key)}\s*=\s*)([^!\n\r]*)(.*)$", re.IGNORECASE | re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", text, count=1)
    if not text.endswith("\n"):
        text += "\n"
    return text + f"  {key} = {value}\n"


def _fortran_bool(value: bool) -> str:
    return ".true." if value else ".false."


def _parse_default_er_values(base_input: Path) -> list[float]:
    nml = read_sfincs_input(base_input)
    er0 = float(nml.group("physicsParameters").get("Er", 0.0))
    scale = max(abs(er0), 1.0)
    return [v * scale for v in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)]


def resolve_er_values(*, er_values_arg: str, base_input: Path) -> list[float]:
    if er_values_arg.strip():
        return [float(v.strip()) for v in er_values_arg.split(",") if v.strip()]
    return _parse_default_er_values(base_input)


def rewrite_trajectory_input(*, base_text: str, er: float, model: TrajectoryModel, fast: bool) -> str:
    text = base_text
    text = _replace_assignment(text, "Er", f"{float(er):.12g}")
    text = _replace_assignment(text, "includeXDotTerm", _fortran_bool(model.include_xdot_term))
    text = _replace_assignment(
        text,
        "includeElectricFieldTermInXiDot",
        _fortran_bool(model.include_electric_field_term_in_xidot),
    )
    text = _replace_assignment(text, "useDKESExBDrift", _fortran_bool(model.use_dkes_exb_drift))
    text = _replace_assignment(text, "magneticDriftScheme", str(int(model.magnetic_drift_scheme)))
    if fast:
        for key, value in (
            ("Ntheta", "5"),
            ("Nzeta", "5"),
            ("Nxi", "3"),
            ("Nx", "3"),
            ("NL", "3"),
            ("solverTolerance", "1e-4"),
        ):
            text = _replace_assignment(text, key, value)
    return text


def collect_sweep_record(*, output_path: Path, model: TrajectoryModel, er: float, er_res: float | None, species_index: int) -> SweepRecord:
    with h5py.File(output_path, "r") as h5:
        pf = np.asarray(h5["particleFlux_vm_psiHat"][...], dtype=np.float64).reshape(-1)
        hf = np.asarray(h5["heatFlux_vm_psiHat"][...], dtype=np.float64).reshape(-1)
        flow = np.asarray(h5["FSABFlow"][...], dtype=np.float64).reshape(-1)
        jhat = np.asarray(h5["FSABjHat"][...], dtype=np.float64).reshape(-1)
    idx = int(species_index)
    return SweepRecord(
        model=model.name,
        label=model.label,
        er=float(er),
        er_over_eres=(float(er) / float(er_res)) if er_res not in {None, 0.0} else None,
        particle_flux_vm_psi_hat=float(pf[idx]),
        heat_flux_vm_psi_hat=float(hf[idx]),
        fsab_flow=float(flow[idx]),
        fsab_jhat=float(jhat[0]),
        output_path=str(output_path),
    )


def write_summary_json(*, summary_path: Path, records: list[SweepRecord]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in records]
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_summary_json(path: Path) -> list[SweepRecord]:
    data = json.loads(path.read_text())
    return [SweepRecord(**row) for row in data]


def plot_er_trajectory_sweep(*, records: list[SweepRecord], out_dir: Path, stem: str, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    series = {
        "particle_flux_vm_psi_hat": (axes[0, 0], r"$\Gamma_{\psi,\mathrm{vm}}$", "particle flux"),
        "heat_flux_vm_psi_hat": (axes[0, 1], r"$Q_{\psi,\mathrm{vm}}$", "heat flux"),
        "fsab_flow": (axes[1, 0], r"$\langle B u_\parallel \rangle$", "FSAB flow"),
        "fsab_jhat": (axes[1, 1], r"$\langle J \cdot B \rangle$", "bootstrap current"),
    }

    use_normalized = any(r.er_over_eres is not None for r in records)
    x_key = "er_over_eres" if use_normalized else "er"
    xlabel = r"$E_r / E_r^{\mathrm{res}}$" if use_normalized else r"$E_r$"

    by_model: dict[str, list[SweepRecord]] = {}
    for rec in records:
        by_model.setdefault(rec.label, []).append(rec)
    for label, model_records in by_model.items():
        model_records = sorted(model_records, key=lambda r: getattr(r, x_key) if getattr(r, x_key) is not None else r.er)
        x = np.asarray([getattr(r, x_key) if getattr(r, x_key) is not None else r.er for r in model_records], dtype=float)
        for field, (ax, ylabel, panel_title) in series.items():
            y = np.asarray([getattr(r, field) for r in model_records], dtype=float)
            ax.plot(x, y, marker="o", label=label)
            ax.set_title(panel_title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, which="both", alpha=0.3)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(title)
    fig.savefig(out_dir / f"{stem}.png", dpi=160)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _run_sweep(*, input_path: Path, work_dir: Path, summary_json: Path, out_dir: Path, er_values: list[float], er_res: float | None, species_index: int, fast: bool) -> None:
    base_text = input_path.read_text()
    records: list[SweepRecord] = []
    for model in TRAJECTORY_MODELS:
        model_dir = work_dir / model.name
        model_dir.mkdir(parents=True, exist_ok=True)
        for er in er_values:
            run_dir = model_dir / f"er_{er:+.6g}".replace("+", "p").replace("-", "m")
            run_dir.mkdir(parents=True, exist_ok=True)
            input_namelist = run_dir / "input.namelist"
            input_namelist.write_text(
                rewrite_trajectory_input(base_text=base_text, er=float(er), model=model, fast=bool(fast))
            )
            output_path = run_dir / "sfincsOutput.h5"
            write_sfincs_jax_output_h5(
                input_namelist=input_namelist,
                output_path=output_path,
                compute_solution=True,
                verbose=False,
            )
            records.append(
                collect_sweep_record(
                    output_path=output_path,
                    model=model,
                    er=float(er),
                    er_res=er_res,
                    species_index=int(species_index),
                )
            )
    write_summary_json(summary_path=summary_json, records=records)
    plot_er_trajectory_sweep(
        records=records,
        out_dir=out_dir,
        stem="sfincs_jax_er_trajectory_sweep",
        title="Trajectory-model sweep versus radial electric field",
    )


def main() -> int:
    args = _parse_args()
    _setup_mpl()
    base_input = args.input if args.input is not None else PRESETS[args.preset]
    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    summary_json = Path(args.summary_json) if args.summary_json is not None else work_dir / "er_trajectory_sweep_summary.json"

    if args.plot_only:
        records = load_summary_json(summary_json)
        plot_er_trajectory_sweep(
            records=records,
            out_dir=out_dir,
            stem="sfincs_jax_er_trajectory_sweep",
            title="Trajectory-model sweep versus radial electric field",
        )
        print(f"Replotted trajectory-model sweep from {summary_json}")
        return 0

    er_values = resolve_er_values(er_values_arg=args.er_values, base_input=base_input)
    _run_sweep(
        input_path=base_input,
        work_dir=work_dir,
        summary_json=summary_json,
        out_dir=out_dir,
        er_values=er_values,
        er_res=args.er_res,
        species_index=int(args.species_index),
        fast=bool(args.fast),
    )
    print(f"Wrote sweep summary to {summary_json}")
    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
