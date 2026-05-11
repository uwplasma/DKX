#!/usr/bin/env python
# ruff: noqa: E402
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

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("This example requires matplotlib. Install with: pip install matplotlib") from exc

from sfincs_jax.ambipolar import AmbipolarSolveResult, solve_ambipolar_from_scan_dir
from sfincs_jax.scans import run_er_scan


DEFAULT_W7X_INPUT = (
    _REPO_ROOT
    / "examples"
    / "sfincs_examples"
    / "filteredW7XNetCDF_2species_magneticDrifts_withEr"
    / "input.namelist"
)

W7X_LITERATURE = (
    "https://sites.fusion.ciemat.es/jlvelasco/files/papers/pablant2020ionroot.pdf",
    "https://sites.fusion.ciemat.es/jlvelasco/files/papers/pablant2018er.pdf",
    "https://www.nature.com/articles/s41586-021-03687-w",
)


@dataclass(frozen=True)
class ScanRunRecord:
    er: float
    scan_variable: float
    radial_current: float
    outputs: dict[str, float]


def load_provenance_json(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def ambipolar_acceptance_gates(
    *,
    ambipolar_result: AmbipolarSolveResult,
    provenance: dict[str, object] | None = None,
    source_artifact_checked_in: bool = False,
    root_er_tolerance: float = 1.0e-9,
    min_abs_current_slope: float = 1.0e-12,
) -> dict[str, bool]:
    roots = np.asarray(ambipolar_result.roots_er, dtype=np.float64)
    er = np.asarray(ambipolar_result.er_values, dtype=np.float64)
    currents = np.asarray(ambipolar_result.radial_currents, dtype=np.float64)
    finite_series = bool(
        er.size >= 2
        and currents.size == er.size
        and np.all(np.isfinite(er))
        and np.all(np.isfinite(currents))
    )
    if finite_series:
        order = np.argsort(er)
        er_sorted = er[order]
        currents_sorted = currents[order]
        distinct_er = bool(np.all(np.diff(er_sorted) > 0.0))
    else:
        er_sorted = np.asarray([], dtype=np.float64)
        currents_sorted = np.asarray([], dtype=np.float64)
        distinct_er = False
    brackets = _zero_current_brackets(er=er_sorted, radial_current=currents_sorted) if distinct_er else []
    finite_roots = bool(roots.size > 0 and np.all(np.isfinite(roots)))
    root_inside_scan_range = bool(
        finite_roots
        and finite_series
        and all(
            float(np.min(er)) - float(root_er_tolerance)
            <= float(root)
            <= float(np.max(er)) + float(root_er_tolerance)
            for root in roots
        )
    )
    matched_brackets = [
        _matching_zero_current_bracket(float(root), brackets=brackets, tolerance=float(root_er_tolerance))
        for root in roots
    ]
    root_consistent = bool(
        finite_roots
        and bool(brackets)
        and all(bracket is not None for bracket in matched_brackets)
    )
    root_slope_resolved = bool(
        root_consistent
        and all(
            abs(float(bracket["local_radial_current_slope"])) >= float(min_abs_current_slope)
            for bracket in matched_brackets
            if bracket is not None
        )
    )
    root_types = [str(value).lower() for value in ambipolar_result.root_types]
    ion_root_candidate = bool(any("ion" in value for value in root_types) or np.any(roots < 0.0))
    provenance = dict(provenance or {})
    required = ("equilibrium_source", "profile_source", "configuration_or_shot", "literature_reference")
    provenance_complete = bool(all(str(provenance.get(key, "")).strip() for key in required))
    numerically_ready = bool(
        finite_series
        and distinct_er
        and bool(brackets)
        and finite_roots
        and root_inside_scan_range
        and root_consistent
        and root_slope_resolved
        and ion_root_candidate
    )
    return {
        "finite_er_current_series": finite_series,
        "distinct_er_scan_points": distinct_er,
        "finite_ambipolar_roots": finite_roots,
        "radial_current_brackets_zero": bool(brackets),
        "root_inside_scanned_er_range": root_inside_scan_range,
        "root_consistent_with_sign_change": root_consistent,
        "ambipolar_root_slope_resolved": root_slope_resolved,
        "ion_root_candidate": ion_root_candidate,
        "provenance_complete": provenance_complete,
        "source_artifact_checked_in": bool(source_artifact_checked_in),
        "numerically_ready_for_literature_artifact": numerically_ready,
        "checked_in_converged_artifact": bool(
            numerically_ready and provenance_complete and source_artifact_checked_in
        ),
        "ready_for_literature_claim": bool(
            numerically_ready and provenance_complete and source_artifact_checked_in
        ),
    }


def ambipolar_deferred_reasons(gates: dict[str, bool]) -> list[dict[str, str]]:
    if bool(gates.get("ready_for_literature_claim")):
        return []
    reason_specs = [
        (
            "nonfinite_or_underresolved_scan",
            "finite_er_current_series",
            "The Er/current scan must contain at least two finite points.",
        ),
        (
            "duplicate_er_scan_points",
            "distinct_er_scan_points",
            "The Er scan points must be distinct before root bracketing is trusted.",
        ),
        (
            "missing_zero_current_bracket",
            "radial_current_brackets_zero",
            "The radial-current scan does not bracket an ambipolar root.",
        ),
        (
            "missing_finite_ambipolar_root",
            "finite_ambipolar_roots",
            "The ambipolar postprocessing did not report a finite root.",
        ),
        (
            "root_outside_scanned_er_range",
            "root_inside_scanned_er_range",
            "At least one reported root lies outside the scanned Er range.",
        ),
        (
            "root_not_supported_by_sign_change",
            "root_consistent_with_sign_change",
            "At least one reported root is not supported by a current sign-change bracket.",
        ),
        (
            "ambipolar_root_slope_unresolved",
            "ambipolar_root_slope_resolved",
            "The local radial-current slope at the accepted root is absent or too small.",
        ),
        (
            "ion_root_candidate_missing",
            "ion_root_candidate",
            "The reported roots do not identify an ion-root candidate.",
        ),
        (
            "incomplete_provenance",
            "provenance_complete",
            "Required W7-X provenance fields are incomplete.",
        ),
        (
            "source_artifact_not_checked_in",
            "source_artifact_checked_in",
            "The source JSON is not a matching checked-in W7-X ambipolar artifact.",
        ),
    ]
    return [
        {"code": code, "gate": gate, "message": message}
        for code, gate, message in reason_specs
        if not bool(gates.get(gate))
    ]


def _zero_current_brackets(
    *,
    er: np.ndarray,
    radial_current: np.ndarray,
) -> list[dict[str, float | int]]:
    brackets: list[dict[str, float | int]] = []
    for idx in range(er.size - 1):
        left_er = float(er[idx])
        right_er = float(er[idx + 1])
        left_current = float(radial_current[idx])
        right_current = float(radial_current[idx + 1])
        crosses = (
            left_current == 0.0
            or right_current == 0.0
            or (left_current < 0.0 < right_current)
            or (right_current < 0.0 < left_current)
        )
        if not crosses:
            continue
        slope = (right_current - left_current) / (right_er - left_er)
        brackets.append(
            {
                "index": int(idx),
                "er_min": float(min(left_er, right_er)),
                "er_max": float(max(left_er, right_er)),
                "local_radial_current_slope": float(slope),
            }
        )
    return brackets


def _matching_zero_current_bracket(
    root: float,
    *,
    brackets: list[dict[str, float | int]],
    tolerance: float,
) -> dict[str, float | int] | None:
    for bracket in brackets:
        if (
            float(bracket["er_min"]) - float(tolerance)
            <= float(root)
            <= float(bracket["er_max"]) + float(tolerance)
        ):
            return bracket
    return None


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_w7x_ambipolar_validation",
        description="Generate a W7-X ambipolar-validation scaffold for sfincs_jax.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_W7X_INPUT, help="Base input.namelist.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=_REPO_ROOT / "examples" / "publication_figures" / "output" / "w7x_ambipolar_validation",
        help="Directory for generated scan inputs and outputs.",
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
        help="Summary JSON path. Defaults to WORK_DIR/w7x_ambipolar_validation_summary.json.",
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file documenting W7-X equilibrium/profile provenance. "
            "Required before the summary can claim literature-grade W7-X validation."
        ),
    )
    parser.add_argument(
        "--er-values",
        default="",
        help="Comma-separated Er values. If omitted, use the input !ss ErMin/ErMax bracket.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=9,
        help="Number of Er points when --er-values is omitted.",
    )
    parser.add_argument(
        "--species-index",
        type=int,
        default=0,
        help="Species index used for figure panels when multiple species are present.",
    )
    parser.add_argument(
        "--n-fine",
        type=int,
        default=500,
        help="Interpolation density for the ambipolar root solve.",
    )
    parser.add_argument(
        "--stem",
        default="sfincs_jax_w7x_ambipolar_validation",
        help="Figure stem (without extension).",
    )
    parser.add_argument(
        "--title",
        default="W7-X ambipolar-validation scaffold",
        help="Figure title.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Apply bounded low-resolution overrides for quicker exploratory scans.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing sfincsOutput.h5 files in the scan directory and only solve missing points.",
    )
    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Populate the scan directory only; skip ambipolar postprocessing and figure generation.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Reuse an existing summary JSON and only regenerate the figure.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Parallel worker processes for scan points.")
    parser.add_argument("--index", type=int, default=None, help="Optional job-array index (0-based).")
    parser.add_argument("--stride", type=int, default=1, help="Stride for job-array slicing.")
    return parser


def _replace_assignment(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(^\s*{re.escape(key)}\s*=\s*)([^!\n\r]*)(.*)$", re.IGNORECASE | re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", text, count=1)
    if not text.endswith("\n"):
        text += "\n"
    return text + f"  {key} = {value}\n"


def _parse_scan_comment_value(text: str, key: str) -> float | None:
    pattern = re.compile(rf"^\s*!ss\s+{re.escape(key)}\s*=\s*([^!\n\r]+)", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        return None
    return float(match.group(1).strip().replace("d", "e").replace("D", "e"))


def resolve_er_values(*, input_path: Path, er_values_arg: str, n_points: int) -> list[float]:
    if er_values_arg.strip():
        return [float(v.strip()) for v in er_values_arg.split(",") if v.strip()]
    text = Path(input_path).read_text()
    er_min = _parse_scan_comment_value(text, "ErMin")
    er_max = _parse_scan_comment_value(text, "ErMax")
    if er_min is not None and er_max is not None:
        return np.linspace(float(er_max), float(er_min), int(n_points), dtype=np.float64).tolist()
    return np.linspace(1.0, -1.0, int(n_points), dtype=np.float64).tolist()


def rewrite_w7x_scan_input(*, base_text: str, fast: bool) -> str:
    text = base_text
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


def build_summary_payload(
    *,
    base_input: Path,
    scan_dir: Path,
    requested_er_values: list[float],
    ambipolar_result: AmbipolarSolveResult,
    fast: bool,
    provenance: dict[str, object] | None = None,
    source_artifact_checked_in: bool = False,
) -> dict[str, object]:
    provenance_payload = dict(provenance or {})
    gates = ambipolar_acceptance_gates(
        ambipolar_result=ambipolar_result,
        provenance=provenance_payload,
        source_artifact_checked_in=bool(source_artifact_checked_in),
    )
    deferred_reasons = ambipolar_deferred_reasons(gates)
    run_records: list[ScanRunRecord] = []
    for idx, er in enumerate(np.asarray(ambipolar_result.er_values, dtype=np.float64), start=0):
        outputs = {
            label: float(ambipolar_result.outputs_by_run[idx, j])
            for j, label in enumerate(ambipolar_result.outputs_labels)
        }
        run_records.append(
            ScanRunRecord(
                er=float(er),
                scan_variable=float(ambipolar_result.var_values[idx]),
                radial_current=float(ambipolar_result.radial_currents[idx]),
                outputs=outputs,
            )
        )
    root_outputs = {
        label: [float(v) for v in np.asarray(values, dtype=np.float64).tolist()]
        for label, values in zip(ambipolar_result.outputs_labels, ambipolar_result.outputs_at_roots, strict=True)
    }
    return {
        "metadata": {
            "schema_version": 1,
            "kind": "w7x_ambipolar_validation_scaffold",
            "base_input": str(base_input.relative_to(_REPO_ROOT)),
            "scan_dir": str(scan_dir.resolve()),
            "source_script": str(Path(__file__).resolve().relative_to(_REPO_ROOT)),
            "fast": bool(fast),
            "requested_er_values": [float(v) for v in requested_er_values],
            "var_name": str(ambipolar_result.var_name),
            "literature": list(W7X_LITERATURE),
            "validation_scope": (
                "w7x_literature_validation"
                if gates["ready_for_literature_claim"]
                else "w7x_literature_candidate_deferred"
                if gates["provenance_complete"]
                else "w7x_like_scaffold"
            ),
            "validation_state": (
                "artifact_backed_literature_ready"
                if gates["ready_for_literature_claim"]
                else "scaffold_deferred"
            ),
            "publication_figure": {
                "claim_status": (
                    "checked_in_converged_artifact"
                    if gates["ready_for_literature_claim"]
                    else "proxy_or_deferred"
                ),
                "artifact_class": (
                    "checked_in_w7x_ambipolar_literature_artifact"
                    if gates["ready_for_literature_claim"]
                    else "generated_w7x_ambipolar_scaffold"
                ),
                "checked_in_converged_artifact": bool(gates["checked_in_converged_artifact"]),
                "ready_for_physics_validation_claim": bool(gates["ready_for_literature_claim"]),
                "manuscript_label": (
                    "checked-in W7-X ambipolar validation"
                    if gates["ready_for_literature_claim"]
                    else "deferred W7-X ambipolar scaffold"
                ),
            },
            "deferred_reason_codes": [str(reason["code"]) for reason in deferred_reasons],
        },
        "provenance": provenance_payload,
        "acceptance_gates": gates,
        "deferred_reasons": deferred_reasons,
        "runs": [asdict(record) for record in run_records],
        "ambipolar": {
            "roots_var": [float(v) for v in np.asarray(ambipolar_result.roots_var, dtype=np.float64).tolist()],
            "roots_er": [float(v) for v in np.asarray(ambipolar_result.roots_er, dtype=np.float64).tolist()],
            "root_types": list(ambipolar_result.root_types),
            "outputs_at_roots": root_outputs,
            "radius_wish": None if ambipolar_result.radius_wish is None else float(ambipolar_result.radius_wish),
            "radius_actual": None if ambipolar_result.radius_actual is None else float(ambipolar_result.radius_actual),
        },
    }


def write_summary_json(*, summary_path: Path, payload: dict[str, object]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_summary_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _select_panel_labels(*, run0_outputs: dict[str, float], prefix: str, species_index: int) -> list[str]:
    labels = [label for label in run0_outputs if label.startswith(prefix)]
    tagged = [label for label in labels if f"(species {int(species_index) + 1})" in label]
    return tagged or labels


def plot_w7x_ambipolar_summary(
    *,
    payload: dict[str, object],
    out_dir: Path,
    stem: str,
    title: str,
    species_index: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = list(payload["runs"])
    if not runs:
        raise ValueError("Need at least one run to plot the ambipolar summary.")
    runs = sorted(runs, key=lambda row: float(row["er"]))
    x = np.asarray([float(row["er"]) for row in runs], dtype=np.float64)
    radial_current = np.asarray([float(row["radial_current"]) for row in runs], dtype=np.float64)
    outputs0 = dict(runs[0]["outputs"])
    heat_labels = _select_panel_labels(run0_outputs=outputs0, prefix="heatFlux", species_index=int(species_index))
    flux_labels = _select_panel_labels(run0_outputs=outputs0, prefix="particleFlux", species_index=int(species_index))
    flow_labels = _select_panel_labels(run0_outputs=outputs0, prefix="FSABFlow", species_index=int(species_index))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    axes[0, 0].plot(x, radial_current, marker="o", label="radial current")
    axes[0, 0].axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    axes[0, 0].set_title("Ambipolarity condition")
    axes[0, 0].set_xlabel(r"$E_r$")
    axes[0, 0].set_ylabel("radial current")

    for label in heat_labels[:2]:
        y = np.asarray([float(row["outputs"][label]) for row in runs], dtype=np.float64)
        axes[0, 1].plot(x, y, marker="o", label=label)
    axes[0, 1].set_title("Heat-flux ordering")
    axes[0, 1].set_xlabel(r"$E_r$")
    axes[0, 1].set_ylabel("heat flux")

    for label in flux_labels[:2]:
        y = np.asarray([float(row["outputs"][label]) for row in runs], dtype=np.float64)
        axes[1, 0].plot(x, y, marker="o", label=label)
    axes[1, 0].set_title("Particle-flux ordering")
    axes[1, 0].set_xlabel(r"$E_r$")
    axes[1, 0].set_ylabel("particle flux")

    for label in flow_labels[:2]:
        y = np.asarray([float(row["outputs"][label]) for row in runs], dtype=np.float64)
        axes[1, 1].plot(x, y, marker="o", label=label)
    if "FSABjHat" in outputs0:
        y = np.asarray([float(row["outputs"]["FSABjHat"]) for row in runs], dtype=np.float64)
        axes[1, 1].plot(x, y, marker="o", label="FSABjHat")
    axes[1, 1].set_title("Flow and current")
    axes[1, 1].set_xlabel(r"$E_r$")
    axes[1, 1].set_ylabel("flow / current")

    roots = np.asarray(payload["ambipolar"]["roots_er"], dtype=np.float64)
    for ax in axes.flat:
        for root in roots:
            ax.axvline(float(root), color="#a23b72", linewidth=1.2, linestyle=":", alpha=0.9)
        ax.grid(True, which="both", alpha=0.3)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    handles2, labels2 = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles + handles2, labels + labels2, loc="upper right")
    fig.suptitle(title)
    fig.savefig(out_dir / f"{stem}.png", dpi=160)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _run_validation(
    *,
    input_path: Path,
    work_dir: Path,
    summary_json: Path,
    out_dir: Path,
    er_values: list[float],
    fast: bool,
    skip_existing: bool,
    scan_only: bool,
    n_fine: int,
    species_index: int,
    stem: str,
    title: str,
    jobs: int,
    index: int | None,
    stride: int,
    provenance_json: Path | None,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    scan_dir = work_dir / "scan"
    template_path = work_dir / "input_template.namelist"
    template_path.write_text(rewrite_w7x_scan_input(base_text=input_path.read_text(), fast=bool(fast)))
    run_er_scan(
        input_namelist=template_path,
        out_dir=scan_dir,
        values=er_values,
        compute_solution=True,
        compute_transport_matrix=False,
        skip_existing=bool(skip_existing),
        jobs=int(jobs),
        index=index,
        stride=int(stride),
    )
    if scan_only:
        return
    ambi = solve_ambipolar_from_scan_dir(scan_dir=scan_dir, write_pickle=True, write_json=True, n_fine=int(n_fine))
    payload = build_summary_payload(
        base_input=input_path,
        scan_dir=scan_dir,
        requested_er_values=er_values,
        ambipolar_result=ambi,
        fast=bool(fast),
        provenance=load_provenance_json(provenance_json),
    )
    write_summary_json(summary_path=summary_json, payload=payload)
    plot_w7x_ambipolar_summary(
        payload=payload,
        out_dir=out_dir,
        stem=stem,
        title=title,
        species_index=int(species_index),
    )


def main(argv: list[str] | None = None) -> int:
    _setup_mpl()
    args = _build_parser().parse_args(argv)
    input_path = Path(args.input).resolve()
    work_dir = Path(args.work_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    summary_json = (
        Path(args.summary_json).resolve()
        if args.summary_json is not None
        else work_dir / "w7x_ambipolar_validation_summary.json"
    )
    if args.scan_only and args.plot_only:
        raise ValueError("Cannot combine --scan-only and --plot-only.")
    if args.plot_only:
        payload = load_summary_json(summary_json)
        plot_w7x_ambipolar_summary(
            payload=payload,
            out_dir=out_dir,
            stem=str(args.stem),
            title=str(args.title),
            species_index=int(args.species_index),
        )
        return 0
    er_values = resolve_er_values(
        input_path=input_path,
        er_values_arg=str(args.er_values),
        n_points=int(args.n_points),
    )
    _run_validation(
        input_path=input_path,
        work_dir=work_dir,
        summary_json=summary_json,
        out_dir=out_dir,
        er_values=er_values,
        fast=bool(args.fast),
        skip_existing=bool(args.skip_existing),
        scan_only=bool(args.scan_only),
        n_fine=int(args.n_fine),
        species_index=int(args.species_index),
        stem=str(args.stem),
        title=str(args.title),
        jobs=int(args.jobs),
        index=int(args.index) if args.index is not None else None,
        stride=int(args.stride),
        provenance_json=Path(args.provenance_json).resolve() if args.provenance_json is not None else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
