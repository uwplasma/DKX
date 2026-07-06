#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from benchmark_case_variants import _parse_profile_events, _profile_stage_durations, _rhs1_preconditioners


_KSP_RE = re.compile(r"ksp_iterations=(?P<iters>\d+)\s+solver=(?P<solver>[a-zA-Z0-9_]+)")
_DENSE_AUTO_MARKERS = (
    "FP RHSMode=1 small system -> using dense solve",
    "FP RHSMode=1 bounded system -> using dense solve",
)
_HOST_DENSE_SHORTCUT_MARKERS = (
    "using host dense shortcut",
    "host dense shortcut on backend=",
)


def _solver_iterations(text: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        match = _KSP_RE.search(line)
        if match is None:
            continue
        rows.append({"solver": match.group("solver").lower(), "iterations": int(match.group("iters"))})
    return rows


def summarize_log(log_path: Path) -> dict[str, object]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    events = _parse_profile_events(text)
    rss_values = [float(event["rss_mb"]) for event in events if event["rss_mb"] is not None]
    stage_durations = _profile_stage_durations(events)
    preconditioners = _rhs1_preconditioners(text)
    return {
        "case": log_path.parent.name,
        "log_path": str(log_path),
        "dense_auto": any(marker in text for marker in _DENSE_AUTO_MARKERS),
        "host_dense_shortcut": any(marker in text for marker in _HOST_DENSE_SHORTCUT_MARKERS),
        "default_krylov": "defaulting to Krylov GMRES" in text,
        "dense_fallback": "rhs1_dense_fallback" in text,
        "sparse_fallback": "rhs1_sparse_precond" in text or "host sparse LU direct fallback" in text,
        "preconditioners": preconditioners,
        "last_preconditioner": preconditioners[-1] if preconditioners else None,
        "ksp_iterations": _solver_iterations(text),
        "profile_peak_rss_mb": max(rss_values) if rss_values else None,
        "profile_stage_durations_s": stage_durations,
        "profile_events_n": len(events),
    }


def _write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# Solver Path Audit\n\n",
        "| Case | Dense auto | Host dense shortcut | Default Krylov | Fallback | Last preconditioner | Peak RSS MB | Slowest profiled stage |\n",
        "| --- | ---: | ---: | ---: | --- | --- | ---: | --- |\n",
    ]
    for row in rows:
        stages = dict(row["profile_stage_durations_s"])
        slowest = "-"
        if stages:
            stage, value = max(stages.items(), key=lambda item: float(item[1]))
            slowest = f"{stage}={float(value):.3f}s"
        fallback = []
        if row["dense_fallback"]:
            fallback.append("dense")
        if row["sparse_fallback"]:
            fallback.append("sparse")
        rss = row["profile_peak_rss_mb"]
        lines.append(
            f"| {row['case']} | {bool(row['dense_auto'])} | {bool(row['host_dense_shortcut'])} | "
            f"{bool(row['default_krylov'])} | "
            f"{','.join(fallback) or '-'} | {row['last_preconditioner'] or '-'} | "
            f"{float(rss):.1f} | {slowest} |\n"
            if rss is not None
            else (
                f"| {row['case']} | {bool(row['dense_auto'])} | {bool(row['host_dense_shortcut'])} | "
                f"{bool(row['default_krylov'])} | "
                f"{','.join(fallback) or '-'} | {row['last_preconditioner'] or '-'} | - | {slowest} |\n"
            )
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize sfincs_jax solver paths from suite logs.")
    parser.add_argument("--suite-root", type=Path, required=True, help="Suite root containing per-case sfincs_jax.log files.")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args()

    logs = sorted(Path(args.suite_root).glob("*/sfincs_jax.log"))
    rows = [summarize_log(path) for path in logs]
    text = json.dumps(rows, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(rows, args.markdown_out)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
