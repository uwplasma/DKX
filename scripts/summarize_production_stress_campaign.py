#!/usr/bin/env python
"""Summarize production stress campaign suite reports.

The production campaign writes one ``suite_report.json`` per case/backend.  The
raw run directories can contain large HDF5, VMEC, and profiler files, so this
script extracts only the compact evidence needed to decide whether a row can be
used for public runtime/parity figures or whether it is blocked by reference
runtime, JAX convergence, memory, or setup cost.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (
    REPO_ROOT / "tests" / "production_stress_cpu_campaign_2026-06-11",
    REPO_ROOT / "tests" / "production_stress_gpu_campaign_2026-06-11",
)
DEFAULT_OUT = REPO_ROOT / "outputs" / "benchmarks" / "production_stress_manifest_2026-06-11" / "campaign_summary.json"


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:  # noqa: BLE001
        return str(path)


def _load_report(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"{path} must contain a JSON list")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise TypeError(f"{path} contains a non-object report row")
        rows.append(item)
    return rows


def _infer_backend(root: Path, report: Path) -> str:
    text = f"{root.name}/{report}".lower()
    if "gpu" in text:
        return "gpu"
    if "cpu" in text:
        return "cpu"
    return "unknown"


def _ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if den == 0.0:
        return None
    return num / den


def _compact_fortran_profile(profile: Any) -> dict[str, Any] | None:
    if not isinstance(profile, dict):
        return None
    compact: dict[str, Any] = {}
    for key in (
        "matrix_shape",
        "matrix_nnz",
        "preconditioner_nnz",
        "which_matrix_nnz",
        "solver_package",
    ):
        if key in profile:
            compact[key] = profile[key]

    timings = profile.get("timings_s")
    if isinstance(timings, dict):
        compact["timings_s"] = {
            key: timings[key]
            for key in sorted(timings)
            if timings.get(key) is not None
        }

    mumps = profile.get("mumps")
    if isinstance(mumps, dict):
        keep = (
            "n",
            "nnz",
            "n_mpi_processes",
            "n_omp_threads",
            "ordering",
            "estimated_factor_entries",
            "estimated_real_factor_space",
            "estimated_integer_factor_space",
            "estimated_max_frontal_size",
            "estimated_elimination_operations",
        )
        compact["mumps"] = {key: mumps[key] for key in keep if key in mumps}
    return compact or None


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    try:
        return float(str(value).replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _last_int(pattern: str, text: str) -> int | None:
    value = _last_float(pattern, text)
    return None if value is None else int(value)


def _parse_live_fortran_profile(log_path: Path) -> dict[str, Any] | None:
    """Extract high-value matrix/MUMPS fields from an in-progress Fortran log."""

    text = log_path.read_text(encoding="utf-8", errors="replace")
    profile: dict[str, Any] = {}
    size_match = re.findall(r"The matrix is\s+(\d+)\s+x\s+(\d+)\s+elements", text, re.IGNORECASE)
    if size_match:
        rows, cols = size_match[-1]
        profile["matrix_shape"] = [int(rows), int(cols)]

    active_which: str | None = None
    which_nnz: dict[str, int] = {}
    for line in text.splitlines():
        which_match = re.search(r"Running populateMatrix with whichMatrix\s*=\s*(\d+)", line, re.IGNORECASE)
        if which_match is not None:
            active_which = which_match.group(1)
            continue
        nnz_match = re.search(r"# of nonzeros in Jacobian(?: preconditioner)? matrix:\s*(\d+)", line, re.IGNORECASE)
        if active_which is not None and nnz_match is not None:
            which_nnz[active_which] = int(nnz_match.group(1))
    if which_nnz:
        profile["which_matrix_nnz"] = which_nnz
        if "1" in which_nnz:
            profile["matrix_nnz"] = which_nnz["1"]
        if "0" in which_nnz:
            profile["preconditioner_nnz"] = which_nnz["0"]

    timings = {
        "preassemble_preconditioner": _last_float(
            r"Time to pre-assemble Jacobian preconditioner matrix:\s*([-+0-9.eEdD]+)", text
        ),
        "assemble_preconditioner": _last_float(
            r"Time to assemble Jacobian preconditioner matrix:\s*([-+0-9.eEdD]+)", text
        ),
        "preassemble_jacobian": _last_float(r"Time to pre-assemble Jacobian matrix:\s*([-+0-9.eEdD]+)", text),
        "assemble_jacobian": _last_float(r"Time to assemble Jacobian matrix:\s*([-+0-9.eEdD]+)", text),
        "metis_reordering": _last_float(r"ELAPSED TIME SPENT IN METIS reordering\s*=\s*([-+0-9.eEdD]+)", text),
        "symbolic_factorization": _last_float(r"ELAPSED TIME IN symbolic factorization\s*=\s*([-+0-9.eEdD]+)", text),
    }
    timings = {key: value for key, value in timings.items() if value is not None}
    if timings:
        profile["timings_s"] = timings

    if "Entering DMUMPS" in text:
        profile["solver_package"] = "mumps"
        mumps = {
            "n": _last_int(r"Entering DMUMPS.*?JOB,\s*N,\s*NNZ\s*=\s*\d+\s+(\d+)\s+\d+", text),
            "nnz": _last_int(r"Entering DMUMPS.*?JOB,\s*N,\s*NNZ\s*=\s*\d+\s+\d+\s+(\d+)", text),
            "n_mpi_processes": _last_int(r"executing #MPI\s*=\s*(\d+)", text),
            "n_omp_threads": _last_int(r"#OMP\s*=\s*(\d+)", text),
            "ordering": "METIS" if re.search(r"Ordering based on METIS", text, re.IGNORECASE) else None,
            "estimated_factor_entries": _last_int(r"Number of entries in factors \(estim\.\)\s*=\s*(\d+)", text),
            "estimated_real_factor_space": _last_int(r"Real space for factors\s+\(estimated\)\s*=\s*(\d+)", text),
            "estimated_integer_factor_space": _last_int(r"Integer space for factors \(estimated\)\s*=\s*(\d+)", text),
            "estimated_max_frontal_size": _last_int(r"Maximum frontal size\s+\(estimated\)\s*=\s*(\d+)", text),
            "estimated_elimination_operations": _last_float(
                r"Operations during elimination \(estim\)\s*=\s*([-+0-9.eEdD]+)", text
            ),
        }
        profile["mumps"] = {key: value for key, value in mumps.items() if value is not None}
    return profile or None


def _compact_row(*, row: dict[str, Any], report: Path, root: Path) -> dict[str, Any]:
    jax_perf_runtime = row.get("jax_logged_elapsed_s")
    if jax_perf_runtime is None:
        jax_perf_runtime = row.get("jax_runtime_s")
    out: dict[str, Any] = {
        "case": row.get("case"),
        "backend": _infer_backend(root, report),
        "source_report": _repo_rel(report),
        "status": row.get("status"),
        "blocker_type": row.get("blocker_type"),
        "message": row.get("message"),
        "final_resolution": row.get("final_resolution"),
        "fortran_runtime_s": row.get("fortran_runtime_s"),
        "jax_runtime_s": row.get("jax_runtime_s"),
        "jax_runtime_s_cold": row.get("jax_runtime_s_cold"),
        "jax_runtime_s_warm": row.get("jax_runtime_s_warm"),
        "jax_logged_elapsed_s": row.get("jax_logged_elapsed_s"),
        "fortran_max_rss_mb": row.get("fortran_max_rss_mb"),
        "jax_max_rss_mb": row.get("jax_max_rss_mb"),
        "runtime_ratio_jax_to_fortran": _ratio(jax_perf_runtime, row.get("fortran_runtime_s")),
        "memory_ratio_jax_to_fortran": _ratio(row.get("jax_max_rss_mb"), row.get("fortran_max_rss_mb")),
        "jax_solver_kinds": row.get("jax_solver_kinds", []),
        "jax_solver_iters_mean": row.get("jax_solver_iters_mean"),
        "n_common_keys": row.get("n_common_keys"),
        "n_mismatch_common": row.get("n_mismatch_common"),
        "strict_n_mismatch_common": row.get("strict_n_mismatch_common"),
        "fortran_profile": _compact_fortran_profile(row.get("fortran_profile")),
    }
    return {key: value for key, value in out.items() if value is not None}


def _find_reports(roots: list[Path]) -> list[tuple[Path, Path]]:
    reports: list[tuple[Path, Path]] = []
    for root in roots:
        if root.is_file() and root.name == "suite_report.json":
            reports.append((root.parent, root))
            continue
        if not root.exists():
            continue
        for report in sorted(root.rglob("suite_report.json")):
            reports.append((root, report))
    return reports


def _find_partial_fortran_logs(roots: list[Path], reports: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    report_dirs = {report.parent.resolve() for _, report in reports}
    logs: list[tuple[Path, Path]] = []
    for root in roots:
        if root.is_file() or not root.exists():
            continue
        for log_path in sorted(root.rglob("fortran_run/sfincs.log")):
            case_out_dir = log_path.parent.parent.parent
            if case_out_dir.resolve() in report_dirs:
                continue
            logs.append((root, log_path))
    return logs


def _partial_row_from_fortran_log(*, root: Path, log_path: Path) -> dict[str, Any]:
    profile = _parse_live_fortran_profile(log_path)
    return {
        "case": log_path.parent.parent.name,
        "backend": _infer_backend(root, log_path),
        "source_report": _repo_rel(log_path),
        "status": "running_or_unreported",
        "blocker_type": "pending_suite_report",
        "message": "Fortran log exists but suite_report.json has not been written yet.",
        "fortran_profile": _compact_fortran_profile(profile),
    }


def build_summary(roots: list[Path]) -> dict[str, Any]:
    reports = _find_reports(roots)
    rows: list[dict[str, Any]] = []
    for root, report in reports:
        for row in _load_report(report):
            rows.append(_compact_row(row=row, report=report, root=root))
    partial_logs = _find_partial_fortran_logs(roots, reports)
    for root, log_path in partial_logs:
        rows.append(_partial_row_from_fortran_log(root=root, log_path=log_path))

    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    blocker_counts = Counter(str(row.get("blocker_type", "unknown")) for row in rows)
    backend_counts = Counter(str(row.get("backend", "unknown")) for row in rows)
    cases_by_status: dict[str, list[str]] = defaultdict(list)
    cases_by_blocker: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        label = f"{row.get('backend')}:{row.get('case')}"
        cases_by_status[str(row.get("status", "unknown"))].append(label)
        cases_by_blocker[str(row.get("blocker_type", "unknown"))].append(label)

    rows_sorted = sorted(rows, key=lambda item: (str(item.get("case")), str(item.get("backend"))))
    return {
        "schema_version": 1,
        "kind": "sfincs_jax_production_stress_campaign_summary",
        "roots": [_repo_rel(root) for root in roots],
        "report_count": len(reports),
        "partial_fortran_log_count": len(partial_logs),
        "row_count": len(rows_sorted),
        "status_counts": dict(sorted(status_counts.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "backend_counts": dict(sorted(backend_counts.items())),
        "cases_by_status": {key: sorted(value) for key, value in sorted(cases_by_status.items())},
        "cases_by_blocker": {key: sorted(value) for key, value in sorted(cases_by_blocker.items())},
        "rows": rows_sorted,
    }


def _write_markdown(summary: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# SFINCS-JAX production stress campaign summary",
        "",
        f"- Reports found: {summary['report_count']}",
        f"- Partial Fortran logs found: {summary['partial_fortran_log_count']}",
        f"- Rows found: {summary['row_count']}",
        f"- Status counts: {json.dumps(summary['status_counts'], sort_keys=True)}",
        f"- Blocker counts: {json.dumps(summary['blocker_counts'], sort_keys=True)}",
        "",
        "| Backend | Case | Status | Blocker | Fortran s | JAX s | JAX/Fortran | Fortran MB | JAX MB | JAX/Fortran MB |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {backend} | {case} | {status} | {blocker} | {ft} | {jt} | {rt} | {fm} | {jm} | {rm} |".format(
                backend=row.get("backend", ""),
                case=row.get("case", ""),
                status=row.get("status", ""),
                blocker=row.get("blocker_type", ""),
                ft=_fmt(row.get("fortran_runtime_s")),
                jt=_fmt(row.get("jax_logged_elapsed_s", row.get("jax_runtime_s"))),
                rt=_fmt(row.get("runtime_ratio_jax_to_fortran")),
                fm=_fmt(row.get("fortran_max_rss_mb")),
                jm=_fmt(row.get("jax_max_rss_mb")),
                rm=_fmt(row.get("memory_ratio_jax_to_fortran")),
            )
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3g}"
    except (TypeError, ValueError):
        return str(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        dest="roots",
        action="append",
        type=Path,
        help="Campaign root or suite_report.json. May be passed multiple times.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON summary output path.")
    parser.add_argument("--markdown-out", type=Path, help="Optional Markdown summary table path.")
    parser.add_argument("--json", action="store_true", help="Print compact summary JSON to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    roots = args.roots if args.roots else list(DEFAULT_ROOTS)
    summary = build_summary(roots)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(summary, args.markdown_out)
    if args.json:
        print(json.dumps({key: summary[key] for key in ("report_count", "row_count", "status_counts")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
