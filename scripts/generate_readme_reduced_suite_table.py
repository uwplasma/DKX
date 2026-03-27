#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
REPORT = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report.json"
REPORT_CPU = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_cpu.json"

BEGIN = "<!-- BEGIN REDUCED_SUITE_TABLE -->"
END = "<!-- END REDUCED_SUITE_TABLE -->"


def _load(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {row["case"]: row for row in data}


def main() -> int:
    if not REPORT_CPU.exists() and not REPORT.exists():
        raise SystemExit(f"Missing report: {REPORT}")

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
