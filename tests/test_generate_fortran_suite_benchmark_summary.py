from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_fortran_suite_benchmark_summary.py"
    spec = importlib.util.spec_from_file_location("generate_fortran_suite_benchmark_summary", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_generate_fortran_suite_benchmark_summary_from_frozen_reports(tmp_path: Path) -> None:
    mod = _load_module()
    repo = Path(__file__).resolve().parents[1]
    out_dir = tmp_path / "figures"
    summary_json = tmp_path / "summary.json"

    rc = mod.main(
        [
            "--cpu-report",
            str(repo / "tests" / "scaled_example_suite_recheck_cpu_frozen_2026-04-23_postkeyfix" / "suite_report.json"),
            "--gpu-report",
            str(
                repo
                / "tests"
                / "scaled_example_suite_recheck_gpu_frozen_2026-04-23_postruntimefix_mem"
                / "suite_report.json"
            ),
            "--out-dir",
            str(out_dir),
            "--summary-json",
            str(summary_json),
            "--stem",
            "suite_benchmark_test",
        ]
    )

    assert rc == 0
    assert (out_dir / "suite_benchmark_test.png").exists()
    assert (out_dir / "suite_benchmark_test.pdf").exists()
    payload = json.loads(summary_json.read_text())
    assert payload["metadata"]["kind"] == "fortran_v3_suite_benchmark_summary"
    assert payload["reports"]["cpu"]["parity_ok_cases"] == 39
    assert payload["reports"]["gpu"]["parity_ok_cases"] == 39
    assert payload["reports"]["cpu"]["strict_mismatch_total"] == 0
    assert payload["reports"]["gpu"]["strict_mismatch_total"] == 0
