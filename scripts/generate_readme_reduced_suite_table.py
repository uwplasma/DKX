#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
REPORT = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report.json"
REPORT_STRICT = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_strict.json"
REPORT_CPU = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_cpu.json"
REPORT_GPU = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_gpu.json"
REPORT_STRICT_CPU = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_strict_cpu.json"
REPORT_STRICT_GPU = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_strict_gpu.json"

BEGIN = "<!-- BEGIN REDUCED_SUITE_TABLE -->"
END = "<!-- END REDUCED_SUITE_TABLE -->"


def _load(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {row["case"]: row for row in data}


def main() -> int:
    if REPORT_CPU.exists():
        rows_cpu = _load(REPORT_CPU)
    else:
        if not REPORT.exists():
            raise SystemExit(f"Missing report: {REPORT}")
        rows_cpu = _load(REPORT)
    if REPORT_GPU.exists():
        rows_gpu = _load(REPORT_GPU)
    else:
        rows_gpu = {}
    if REPORT_STRICT_CPU.exists():
        rows_strict = _load(REPORT_STRICT_CPU)
    elif REPORT_STRICT.exists():
        rows_strict = _load(REPORT_STRICT)
    else:
        rows_strict = {}

    practical_ok = sum(1 for row in rows_cpu.values() if str(row.get("status", "")).strip() == "parity_ok")
    strict_ok = sum(
        1
        for case, row in rows_cpu.items()
        if int(rows_strict.get(case, row).get("n_mismatch_common", 0)) == 0
        and str(row.get("status", "")).strip() in {"", "parity_ok", "parity_mismatch"}
    )
    total = len(rows_cpu)
    gpu_rows = len(rows_gpu)
    block_lines = [
        "Archived reduced-suite reports are kept in:",
        "",
        "- `tests/reduced_upstream_examples/suite_report.json`",
        "- `tests/reduced_upstream_examples/suite_report_strict.json`",
        "- `tests/reduced_upstream_examples/suite_report_cpu.json`",
        "- `tests/reduced_upstream_examples/suite_report_gpu.json`",
        "- `docs/_generated/reduced_upstream_suite_status.rst`",
        "- `docs/_generated/reduced_upstream_suite_status_strict.rst`",
        "",
        f"Historical reduced-suite snapshot counts: CPU practical `parity_ok={practical_ok}/{total}`, strict `parity_ok={strict_ok}/{total}`.",
        f"Historical reduced-suite GPU rows archived: `{gpu_rows}`.",
        "",
        "Use these only for historical debugging and comparison against older milestones.",
        "The release-facing parity status for `main` is the full example-suite table below.",
    ]

    readme = README.read_text()
    if BEGIN not in readme or END not in readme:
        raise SystemExit("README markers not found.")
    prefix, rest = readme.split(BEGIN, 1)
    _table, suffix = rest.split(END, 1)
    new_block = "\n".join([BEGIN, *block_lines, END])
    README.write_text(prefix + new_block + suffix)
    print("Updated README reduced-suite table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
