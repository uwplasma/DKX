#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report.json"
REPORT_CPU = REPO_ROOT / "tests" / "reduced_upstream_examples" / "suite_report_cpu.json"
OUT = REPO_ROOT / "docs" / "_generated" / "reduced_suite_archive_note.rst"

def main() -> int:
    if not REPORT_CPU.exists() and not REPORT.exists():
        raise SystemExit(f"Missing report: {REPORT}")
    REPORT_PATHS = [
        "tests/reduced_upstream_examples/suite_report.json",
        "tests/reduced_upstream_examples/suite_report_strict.json",
        "tests/reduced_upstream_examples/suite_report_cpu.json",
        "tests/reduced_upstream_examples/suite_report_gpu.json",
        "docs/_generated/reduced_upstream_suite_status.rst",
        "docs/_generated/reduced_upstream_suite_status_strict.rst",
    ]
    lines = [
        "Archived reduced-suite reports are kept in:",
        "",
        *[f"- ``{path}``" for path in REPORT_PATHS],
        "",
        "Use these artifacts for historical debugging, fast local triage, and comparison against older milestones.",
        "The release-facing parity status for ``main`` is the full example-suite audit documented in ``README.md`` and ``docs/parity.rst``.",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Updated {OUT}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
