#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import selectors
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

import h5py
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS = REPO_ROOT / "examples" / "sfincs_examples" / "utils"
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
        "--nuprime-min",
        type=float,
        default=0.1,
        help="Minimum normalized collisionality nu' for generated scan inputs.",
    )
    parser.add_argument(
        "--nuprime-max",
        type=float,
        default=10.0,
        help="Maximum normalized collisionality nu' for generated scan inputs.",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=None,
        help="Number of collisionality scan points. Defaults to 4 with --fast and 7 otherwise.",
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
    parser.add_argument(
        "--transport-workers",
        type=int,
        default=1,
        help=(
            "Number of process workers for independent whichRHS transport solves. "
            "Use 2 on dual-GPU nodes to run high-nu FP/PAS pilots with one RHS worker per GPU."
        ),
    )
    parser.add_argument(
        "--transport-parallel-backend",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Backend for --transport-workers: auto/cpu use CPU workers; gpu pins workers to visible CUDA devices.",
    )
    parser.add_argument(
        "--require-residuals",
        action="store_true",
        help=(
            "Require residual diagnostics in existing sfincsOutput.h5 files before "
            "reusing them with --skip-existing or --plot-only."
        ),
    )
    parser.add_argument(
        "--max-transport-residual",
        type=float,
        default=1.0e-6,
        help="Maximum accepted transport residual when --require-residuals is used.",
    )
    parser.add_argument(
        "--max-transport-relative-residual",
        type=float,
        default=1.0e-6,
        help=(
            "Maximum accepted transport residual divided by RHS norm when "
            "--require-residuals is used and relative diagnostics are present."
        ),
    )
    parser.add_argument(
        "--transport-sparse-direct-max",
        type=int,
        default=0,
        help=(
            "Optional SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX override for publication "
            "transport scans. Use a positive bounded value for high-nu campaigns."
        ),
    )
    parser.add_argument(
        "--transport-maxiter",
        type=int,
        default=0,
        help="Optional SFINCS_JAX_TRANSPORT_MAXITER override for transport scan solves.",
    )
    return parser.parse_args()


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_s: float | None,
    label: str,
    extra_env: dict[str, str] | None = None,
) -> None:
    print(f"[{label}] cwd={cwd}")
    print(f"[{label}] cmd={' '.join(cmd)}")
    log_path = cwd / f"{label}.log"
    print(f"[{label}] log={log_path}")
    sys.stdout.flush()
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLCONFIGDIR", str(cwd / ".mplconfig"))
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    t0 = time.perf_counter()
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        try:
            while True:
                remaining_timeout = None
                if timeout_s is not None:
                    remaining_timeout = float(timeout_s) - float(time.perf_counter() - t0)
                    if remaining_timeout <= 0.0:
                        proc.kill()
                        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout_s)
                wait_timeout = 1.0 if remaining_timeout is None else min(1.0, max(0.0, remaining_timeout))
                events = selector.select(wait_timeout)
                if not events:
                    if proc.poll() is not None:
                        break
                    continue
                saw_eof = False
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line == "":
                        saw_eof = True
                        continue
                    print(line, end="")
                    log.write(line)
                    log.flush()
                if saw_eof and proc.poll() is not None:
                    break
            remainder = proc.stdout.read()
            if remainder:
                print(remainder, end="")
                log.write(remainder)
                log.flush()
        finally:
            selector.close()
        retcode = proc.wait()
        if retcode != 0:
            raise subprocess.CalledProcessError(retcode, cmd)
    print(f"[{label}] completed elapsed={time.perf_counter() - t0:.1f}s")
    sys.stdout.flush()


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


def _transport_residuals_from_h5(h5_path: Path) -> np.ndarray | None:
    with h5py.File(h5_path, "r") as f:
        if "transportResidualNorms" in f:
            return np.asarray(f["transportResidualNorms"][()], dtype=np.float64).reshape((-1,))
        if "transportMaxResidualNorm" in f:
            return np.asarray([float(np.asarray(f["transportMaxResidualNorm"][()]))], dtype=np.float64)
    return None


def _transport_relative_residuals_from_h5(h5_path: Path) -> np.ndarray | None:
    with h5py.File(h5_path, "r") as f:
        if "transportRelativeResidualNorms" in f:
            return np.asarray(f["transportRelativeResidualNorms"][()], dtype=np.float64).reshape((-1,))
        if "transportMaxRelativeResidualNorm" in f:
            return np.asarray([float(np.asarray(f["transportMaxRelativeResidualNorm"][()]))], dtype=np.float64)
    return None


def _output_passes_transport_quality(
    h5_path: Path,
    *,
    require_residuals: bool,
    max_residual: float,
    max_relative_residual: float = 1.0e-6,
) -> bool:
    if not h5_path.exists():
        return False
    if not bool(require_residuals):
        return True
    try:
        residuals = _transport_residuals_from_h5(h5_path)
        relative_residuals = _transport_relative_residuals_from_h5(h5_path)
    except OSError:
        return False
    if residuals is None or residuals.size == 0:
        return False
    if not np.all(np.isfinite(residuals)):
        return False
    if float(np.max(np.abs(residuals))) > float(max_residual):
        return False
    if relative_residuals is not None and relative_residuals.size > 0:
        if not np.all(np.isfinite(relative_residuals)):
            return False
        if float(np.max(np.abs(relative_residuals))) > float(max_relative_residual):
            return False
    return True


def _collect_transport_quality(
    work_dir: Path,
    *,
    max_residual: float,
    max_relative_residual: float = 1.0e-6,
) -> dict[str, object]:
    output_count = 0
    residual_count = 0
    missing_residual_count = 0
    residual_values: list[float] = []
    relative_residual_values: list[float] = []
    failed_outputs: list[str] = []
    for sub in sorted(work_dir.iterdir()):
        if not sub.is_dir():
            continue
        h5_path = sub / "sfincsOutput.h5"
        if not h5_path.exists():
            continue
        output_count += 1
        try:
            residuals = _transport_residuals_from_h5(h5_path)
            relative_residuals = _transport_relative_residuals_from_h5(h5_path)
        except OSError:
            missing_residual_count += 1
            failed_outputs.append(sub.name)
            continue
        if residuals is None or residuals.size == 0:
            missing_residual_count += 1
            failed_outputs.append(sub.name)
            continue
        residual_count += 1
        residual_values.extend(float(v) for v in residuals.reshape((-1,)))
        if relative_residuals is not None:
            relative_residual_values.extend(float(v) for v in relative_residuals.reshape((-1,)))
        if (not np.all(np.isfinite(residuals))) or float(np.max(np.abs(residuals))) > float(max_residual):
            failed_outputs.append(sub.name)
        elif relative_residuals is not None and (
            (not np.all(np.isfinite(relative_residuals)))
            or float(np.max(np.abs(relative_residuals))) > float(max_relative_residual)
        ):
            failed_outputs.append(sub.name)
    finite = np.asarray([v for v in residual_values if np.isfinite(v)], dtype=np.float64)
    finite_relative = np.asarray([v for v in relative_residual_values if np.isfinite(v)], dtype=np.float64)
    max_seen = float(np.max(np.abs(finite))) if finite.size else None
    max_relative_seen = float(np.max(np.abs(finite_relative))) if finite_relative.size else None
    return {
        "output_count": int(output_count),
        "residual_output_count": int(residual_count),
        "missing_residual_output_count": int(missing_residual_count),
        "max_transport_residual_norm": max_seen,
        "max_transport_residual_allowed": float(max_residual),
        "max_transport_relative_residual_norm": max_relative_seen,
        "max_transport_relative_residual_allowed": float(max_relative_residual),
        "residual_gate_passed": bool(output_count > 0 and not failed_outputs),
        "failed_outputs": failed_outputs,
    }


def _has_transport_outputs(work_dir: Path) -> bool:
    return any(work_dir.glob("*/sfincsOutput.h5"))


def _scan_values(*, min_value: float, max_value: float, n_points: int, scale: str = "log") -> tuple[float, ...]:
    n_points = int(n_points)
    if n_points < 1:
        return ()
    if n_points == 1:
        return (float(min_value),)
    scale_kind = str(scale).strip().lower()
    if scale_kind == "log":
        sign = -1.0 if float(min_value) < 0.0 else 1.0
        values = np.exp(np.linspace(np.log(abs(float(min_value))), np.log(abs(float(max_value))), n_points))
        return tuple(float(sign * value) for value in values)
    if scale_kind == "linear":
        return tuple(float(value) for value in np.linspace(float(min_value), float(max_value), n_points))
    raise ValueError(f"Unsupported scan scale {scale!r}.")


def _expected_scan_subdirs(
    *,
    scan_variable: str,
    min_value: float,
    max_value: float,
    n_points: int,
    scale: str = "log",
) -> tuple[str, ...]:
    values = _scan_values(min_value=min_value, max_value=max_value, n_points=n_points, scale=scale)
    return tuple(f"{scan_variable}_{value:.4g}" for value in values)


def _scan_dir_complete(
    work_dir: Path,
    *,
    expected_subdirs: tuple[str, ...],
    require_residuals: bool = False,
    max_residual: float = 1.0e-6,
    max_relative_residual: float = 1.0e-6,
) -> bool:
    if not expected_subdirs:
        return False
    return all(
        _output_passes_transport_quality(
            work_dir / name / "sfincsOutput.h5",
            require_residuals=bool(require_residuals),
            max_residual=float(max_residual),
            max_relative_residual=float(max_relative_residual),
        )
        for name in expected_subdirs
    )


def _require_complete_plot_only_scan(
    work_dir: Path,
    *,
    case_name: str,
    collision_operator: int,
    expected_subdirs: tuple[str, ...],
    require_residuals: bool = False,
    max_residual: float = 1.0e-6,
    max_relative_residual: float = 1.0e-6,
) -> None:
    if _scan_dir_complete(
        work_dir,
        expected_subdirs=expected_subdirs,
        require_residuals=bool(require_residuals),
        max_residual=float(max_residual),
        max_relative_residual=float(max_relative_residual),
    ):
        return
    found = sum(1 for _ in work_dir.glob("*/sfincsOutput.h5"))
    expected = len(expected_subdirs)
    residual_msg = " with residual diagnostics" if require_residuals else ""
    raise RuntimeError(
        "--plot-only requires a complete scan before publication summaries are "
        f"rewritten{residual_msg}: case={case_name}, collisionOperator={collision_operator}, "
        f"found={found}/{expected}, work_dir={work_dir}"
    )


def _prune_incomplete_scan_dirs(
    work_dir: Path,
    *,
    expected_subdirs: tuple[str, ...],
    require_residuals: bool = False,
    max_residual: float = 1.0e-6,
    max_relative_residual: float = 1.0e-6,
) -> None:
    for name in expected_subdirs:
        subdir = work_dir / name
        if subdir.exists() and not _output_passes_transport_quality(
            subdir / "sfincsOutput.h5",
            require_residuals=bool(require_residuals),
            max_residual=float(max_residual),
            max_relative_residual=float(max_relative_residual),
        ):
            shutil.rmtree(subdir, ignore_errors=True)


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


def _transport_scan_env(
    *,
    transport_workers: int,
    transport_parallel_backend: str,
    transport_sparse_direct_max: int = 0,
    transport_maxiter: int = 0,
    abort_max_residual: float = 0.0,
    abort_max_relative_residual: float = 0.0,
) -> dict[str, str]:
    """Return scan-run environment overrides for the fast explicit transport path."""
    env = {
        "SFINCS_JAX_IMPLICIT_SOLVE": "0",
        "SFINCS_JAX_WRITE_SOLVER_DIAGNOSTICS": "1",
    }
    workers = max(1, int(transport_workers))
    backend = str(transport_parallel_backend).strip().lower()
    if workers > 1:
        env["SFINCS_JAX_TRANSPORT_PARALLEL"] = "process"
        env["SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS"] = str(workers)
        if backend in {"cpu", "gpu"}:
            env["SFINCS_JAX_TRANSPORT_PARALLEL_BACKEND"] = backend
    if int(transport_sparse_direct_max) > 0:
        env["SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX"] = str(int(transport_sparse_direct_max))
    if int(transport_maxiter) > 0:
        env["SFINCS_JAX_TRANSPORT_MAXITER"] = str(int(transport_maxiter))
    if float(abort_max_residual) > 0.0:
        env["SFINCS_JAX_TRANSPORT_ABORT_MAX_RESIDUAL"] = f"{float(abort_max_residual):.16g}"
    if float(abort_max_relative_residual) > 0.0:
        env["SFINCS_JAX_TRANSPORT_ABORT_MAX_RELATIVE_RESIDUAL"] = (
            f"{float(abort_max_relative_residual):.16g}"
        )
    return env


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
    transport_workers: int,
    transport_parallel_backend: str,
    require_residuals: bool = False,
    max_transport_residual: float = 1.0e-6,
    max_transport_relative_residual: float = 1.0e-6,
    transport_sparse_direct_max: int = 0,
    transport_maxiter: int = 0,
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
        "implicit_solve": False,
        "transport_workers": int(transport_workers),
        "transport_parallel_backend": str(transport_parallel_backend),
        "write_solver_diagnostics": True,
        "require_residuals": bool(require_residuals),
        "max_transport_residual": float(max_transport_residual),
        "max_transport_relative_residual": float(max_transport_relative_residual),
        "transport_sparse_direct_max": int(transport_sparse_direct_max),
        "transport_maxiter": int(transport_maxiter),
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
    transport_workers = max(1, int(args.transport_workers))
    transport_parallel_backend = str(args.transport_parallel_backend)
    require_residuals = bool(args.require_residuals)
    max_transport_residual = float(args.max_transport_residual)
    max_transport_relative_residual = float(args.max_transport_relative_residual)
    transport_sparse_direct_max = max(0, int(args.transport_sparse_direct_max))
    transport_maxiter = max(0, int(args.transport_maxiter))
    scan_env = _transport_scan_env(
        transport_workers=transport_workers,
        transport_parallel_backend=transport_parallel_backend,
        transport_sparse_direct_max=transport_sparse_direct_max,
        transport_maxiter=transport_maxiter,
        abort_max_residual=max_transport_residual if require_residuals else 0.0,
        abort_max_relative_residual=max_transport_relative_residual if require_residuals else 0.0,
    )

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

    n_points = int(args.n_points) if args.n_points is not None else (4 if fast else 7)
    if n_points < 1:
        raise ValueError("--n-points must be at least 1.")
    nuprime_min = float(args.nuprime_min)
    nuprime_max = float(args.nuprime_max)
    if nuprime_min <= 0.0 or nuprime_max <= 0.0 or nuprime_max < nuprime_min:
        raise ValueError("--nuprime-min and --nuprime-max must be positive with max >= min.")

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
    fig1_quality: dict[str, dict[str, object]] = {}

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
            expected_subdirs = _expected_scan_subdirs(
                scan_variable="nu_n",
                min_value=nu_n_min,
                max_value=nu_n_max,
                n_points=n_points,
                scale="log",
            )
            case_dir = work_dir / f"{cfg.name}_co{collision_operator}"
            case_dir.mkdir(parents=True, exist_ok=True)
            if plot_only:
                _require_complete_plot_only_scan(
                    case_dir,
                    case_name=cfg.name,
                    collision_operator=collision_operator,
                    expected_subdirs=expected_subdirs,
                    require_residuals=require_residuals,
                    max_residual=max_transport_residual,
                    max_relative_residual=max_transport_relative_residual,
                )
            else:
                if skip_existing:
                    _prune_incomplete_scan_dirs(
                        case_dir,
                        expected_subdirs=expected_subdirs,
                        require_residuals=require_residuals,
                        max_residual=max_transport_residual,
                        max_relative_residual=max_transport_relative_residual,
                    )
                    should_run = not _scan_dir_complete(
                        case_dir,
                        expected_subdirs=expected_subdirs,
                        require_residuals=require_residuals,
                        max_residual=max_transport_residual,
                        max_relative_residual=max_transport_relative_residual,
                    )
                else:
                    should_run = True
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
                        extra_env=scan_env,
                    )
            if _has_transport_outputs(case_dir):
                quality = _collect_transport_quality(
                    case_dir,
                    max_residual=max_transport_residual,
                    max_relative_residual=max_transport_relative_residual,
                )
                if require_residuals and not bool(quality["residual_gate_passed"]):
                    raise RuntimeError(
                        "transport residual gate failed: "
                        f"case={cfg.name} collisionOperator={collision_operator} quality={quality}"
                    )
                fig1_quality[label] = quality
                fig1_data[label] = _collect_transport_matrix(case_dir)

        if fig1_data:
            lhd_summary_path = summary_dir / f"lhd_collisionality{'_fast' if fast else ''}_summary.json"
            lhd_metadata = _summary_metadata(
                case="lhd",
                fast=fast,
                n_points=n_points,
                nuprime_min=nuprime_min,
                nuprime_max=nuprime_max,
                work_dir=work_dir,
                summary_path=lhd_summary_path,
                base_input=lhd.base_input,
                labels_to_collision_operator={label: operator for label, operator in (("Fokker-Planck", 0), ("PAS", 1)) if operator in selected_collision_operators},
                transport_workers=transport_workers,
                transport_parallel_backend=transport_parallel_backend,
                require_residuals=require_residuals,
                max_transport_residual=max_transport_residual,
                max_transport_relative_residual=max_transport_relative_residual,
                transport_sparse_direct_max=transport_sparse_direct_max,
                transport_maxiter=transport_maxiter,
            )
            lhd_metadata["quality_by_label"] = fig1_quality
            write_transport_scan_summary_json(
                lhd_summary_path,
                fig1_data,
                metadata=lhd_metadata,
            )

        if not scan_only and fig1_data:
            _plot_matrix_elements(
                out_path=out_dir / "sfincs_jax_fig1_lhd_collisionality.png",
                title="LHD collisionality scan (sfincs_jax)",
                datasets=fig1_data,
            )

    # Figure 2 (W7-X)
    fig2_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fig2_quality: dict[str, dict[str, object]] = {}
    if case in ("w7x", "all"):
        for collision_operator, label in [(0, "Fokker-Planck"), (1, "PAS")]:
            if collision_operator not in selected_collision_operators:
                continue
            nu_n_min = nuprime_min * w7x.nuprime_factor
            nu_n_max = nuprime_max * w7x.nuprime_factor
            expected_subdirs = _expected_scan_subdirs(
                scan_variable="nu_n",
                min_value=nu_n_min,
                max_value=nu_n_max,
                n_points=n_points,
                scale="log",
            )
            case_dir = work_dir / f"w7x_co{collision_operator}"
            case_dir.mkdir(parents=True, exist_ok=True)
            if plot_only:
                _require_complete_plot_only_scan(
                    case_dir,
                    case_name=w7x.name,
                    collision_operator=collision_operator,
                    expected_subdirs=expected_subdirs,
                    require_residuals=require_residuals,
                    max_residual=max_transport_residual,
                    max_relative_residual=max_transport_relative_residual,
                )
            else:
                if skip_existing:
                    _prune_incomplete_scan_dirs(
                        case_dir,
                        expected_subdirs=expected_subdirs,
                        require_residuals=require_residuals,
                        max_residual=max_transport_residual,
                        max_relative_residual=max_transport_relative_residual,
                    )
                    should_run = not _scan_dir_complete(
                        case_dir,
                        expected_subdirs=expected_subdirs,
                        require_residuals=require_residuals,
                        max_residual=max_transport_residual,
                        max_relative_residual=max_transport_relative_residual,
                    )
                else:
                    should_run = True
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
                        extra_env=scan_env,
                    )
            if _has_transport_outputs(case_dir):
                quality = _collect_transport_quality(
                    case_dir,
                    max_residual=max_transport_residual,
                    max_relative_residual=max_transport_relative_residual,
                )
                if require_residuals and not bool(quality["residual_gate_passed"]):
                    raise RuntimeError(
                        "transport residual gate failed: "
                        f"case={w7x.name} collisionOperator={collision_operator} quality={quality}"
                    )
                fig2_quality[label] = quality
                fig2_data[label] = _collect_transport_matrix(case_dir)

        if fig2_data:
            w7x_summary_path = summary_dir / f"w7x_collisionality{'_fast' if fast else ''}_summary.json"
            w7x_metadata = _summary_metadata(
                case="w7x",
                fast=fast,
                n_points=n_points,
                nuprime_min=nuprime_min,
                nuprime_max=nuprime_max,
                work_dir=work_dir,
                summary_path=w7x_summary_path,
                base_input=w7x.base_input,
                labels_to_collision_operator={label: operator for label, operator in (("Fokker-Planck", 0), ("PAS", 1)) if operator in selected_collision_operators},
                transport_workers=transport_workers,
                transport_parallel_backend=transport_parallel_backend,
                require_residuals=require_residuals,
                max_transport_residual=max_transport_residual,
                max_transport_relative_residual=max_transport_relative_residual,
                transport_sparse_direct_max=transport_sparse_direct_max,
                transport_maxiter=transport_maxiter,
            )
            w7x_metadata["quality_by_label"] = fig2_quality
            write_transport_scan_summary_json(
                w7x_summary_path,
                fig2_data,
                metadata=w7x_metadata,
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
