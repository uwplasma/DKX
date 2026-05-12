from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_JSON = (
    REPO_ROOT
    / "examples"
    / "publication_figures"
    / "artifacts"
    / "sfincs_jax_fortran_suite_benchmark_summary.json"
)
FIGURE_ROOT = REPO_ROOT / "docs" / "_static" / "figures" / "paper"
BENCHMARK_STEM = "sfincs_jax_fortran_suite_benchmark_summary"


def _summary() -> dict[str, object]:
    return json.loads(SUMMARY_JSON.read_text())


def _ratio_token(payload: dict[str, object], backend: str, key: str, digits: int) -> str:
    reports = payload["reports"]
    assert isinstance(reports, dict)
    report = reports[backend]
    assert isinstance(report, dict)
    ratio_summary = report[key]
    assert isinstance(ratio_summary, dict)
    return f"{float(ratio_summary['median']):.{digits}f}x"


def test_benchmark_doc_ratio_claims_match_checked_summary() -> None:
    payload = _summary()
    cpu_cold = _ratio_token(payload, "cpu", "cold_runtime_ratio_summary", 3)
    gpu_cold = _ratio_token(payload, "gpu", "cold_runtime_ratio_summary", 3)
    cpu_active_memory = _ratio_token(payload, "cpu", "active_memory_ratio_summary", 2)
    gpu_active_memory = _ratio_token(payload, "gpu", "active_memory_ratio_summary", 2)
    cpu_process_memory = _ratio_token(payload, "cpu", "memory_ratio_summary", 2)
    gpu_process_memory = _ratio_token(payload, "gpu", "memory_ratio_summary", 2)

    fortran_comparison = (REPO_ROOT / "docs" / "fortran_comparison.rst").read_text()
    assert cpu_cold in fortran_comparison
    assert gpu_cold in fortran_comparison
    assert cpu_active_memory in fortran_comparison
    assert gpu_active_memory in fortran_comparison

    performance = (REPO_ROOT / "docs" / "performance.rst").read_text()
    assert cpu_cold in performance
    assert gpu_cold in performance
    assert cpu_active_memory in performance
    assert gpu_active_memory in performance
    assert cpu_process_memory in performance
    assert gpu_process_memory in performance


def test_benchmark_artifacts_and_references_are_release_scoped() -> None:
    payload = _summary()
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["kind"] == "fortran_v3_suite_benchmark_summary"
    assert metadata["min_fortran_runtime_s"] == 10.0
    assert len(metadata["excluded_low_fortran_runtime_cases"]) == 15

    for suffix in (".png", ".pdf"):
        artifact = FIGURE_ROOT / f"{BENCHMARK_STEM}{suffix}"
        assert artifact.exists(), artifact
        assert artifact.stat().st_size > 1024, artifact

    checked_paths = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "fortran_comparison.rst",
        REPO_ROOT / "docs" / "index.rst",
        REPO_ROOT / "docs" / "paper_figures.rst",
        REPO_ROOT / "docs" / "performance.rst",
        REPO_ROOT / "docs" / "validation_matrix.rst",
    ]
    for path in checked_paths:
        text = path.read_text()
        assert f"{BENCHMARK_STEM}.png" in text
        assert "reference-runtime-window" in text
        assert "production-scale subset" not in text
        assert "plotted production-scale case" not in text
