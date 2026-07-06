from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_JSON = (
    REPO_ROOT
    / "examples"
    / "publication_figures"
    / "artifacts"
    / "sfincs_jax_fortran_suite_benchmark_summary.json"
)
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
FIGURE_ROOT = REPO_ROOT / "docs" / "_static" / "figures" / "paper"
BENCHMARK_STEM = "sfincs_jax_fortran_suite_benchmark_summary"
PUBLIC_STANDALONE_DOCS = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "sfincs_jax" / "README.md",
    REPO_ROOT / "examples" / "README.md",
    REPO_ROOT / "docs" / "index.rst",
    REPO_ROOT / "docs" / "installation.rst",
    REPO_ROOT / "docs" / "examples.rst",
    REPO_ROOT / "docs" / "feature_matrix.rst",
    REPO_ROOT / "docs" / "fortran_comparison.rst",
    REPO_ROOT / "docs" / "usage.rst",
    REPO_ROOT / "docs" / "method.rst",
    REPO_ROOT / "docs" / "geometry.rst",
    REPO_ROOT / "docs" / "inputs.rst",
    REPO_ROOT / "docs" / "outputs.rst",
    REPO_ROOT / "docs" / "performance.rst",
    REPO_ROOT / "docs" / "parity.rst",
    REPO_ROOT / "docs" / "validation_matrix.rst",
    REPO_ROOT / "docs" / "optimization.rst",
    REPO_ROOT / "docs" / "testing.rst",
    REPO_ROOT / "docs" / "source_map.rst",
    REPO_ROOT / "docs" / "api.rst",
]
PUBLIC_STALE_FRAGMENTS = (
    "On the current",
    "current main branch",
    "current ``main``",
    "Current release snapshot",
    "new version",
    "new benchmarks",
    "README-facing runtime/memory rows",
    "The production benchmark manifest",
    "production benchmark manifest",
    "not replacements for the production-resolution gates",
    "current-tip",
    "Recent current-tip",
    "earlier releases",
    "prior release",
    "latest guarded audit",
    "latest root drift",
)
PUBLIC_STALE_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(word)}\b")
    for word in ("now", "older", "newer", "previous", "currently")
)
DOC_TREE_STALE_FRAGMENTS = (
    "On the current main branch",
    "current main branch",
    "new version",
    "new benchmarks",
    "README-facing runtime/memory rows",
    "The production benchmark manifest",
    "production benchmark manifest",
    "not replacements for the production-resolution gates",
    "not a public performance row",
)


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


def _readme_benchmark_table_rows() -> list[str]:
    readme = (REPO_ROOT / "README.md").read_text().splitlines()
    try:
        start = readme.index("Full per-case runtime / memory table:")
    except ValueError as exc:  # pragma: no cover - assertion gives clearer failure
        raise AssertionError("README is missing the benchmark table heading") from exc
    rows: list[str] = []
    for line in readme[start + 1 :]:
        if rows and not line:
            break
        if line.startswith("| `"):
            rows.append(line)
    return rows


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
        assert "audited reduced example suite" not in text
        assert "production-resolution benchmark tier is now being used for public runtime/memory claims" not in text


def test_readme_suite_counts_and_filtered_table_scope_match_summary() -> None:
    payload = _summary()
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    source_counts = metadata["source_case_counts"]
    reported_counts = metadata["reported_case_counts"]
    assert isinstance(source_counts, dict)
    assert isinstance(reported_counts, dict)

    readme = (REPO_ROOT / "README.md").read_text()
    for label, backend in (
        ("Practical status counts", "cpu"),
        ("Strict status counts", "cpu"),
        ("GPU practical status counts", "gpu"),
        ("GPU strict status counts", "gpu"),
    ):
        expected_source_count = int(source_counts[backend])
        assert f"- {label}: `parity_ok={expected_source_count}`" in readme

    rows = _readme_benchmark_table_rows()
    assert len(rows) == int(reported_counts["cpu"]) == int(reported_counts["gpu"])

    excluded_cases = metadata["excluded_low_fortran_runtime_cases"]
    assert isinstance(excluded_cases, list)
    excluded_case_names = {str(row["case"]) for row in excluded_cases if isinstance(row, dict)}
    table_case_names = {line.split("`", maxsplit=2)[1] for line in rows}
    assert table_case_names.isdisjoint(excluded_case_names)
    assert "The public runtime/memory table is restricted" in readme


def test_readme_is_self_contained_not_branch_history() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    stale_fragments = (
        "On the current main branch",
        "current main branch",
        "new version",
        "new benchmarks",
        "README-facing runtime/memory rows",
        "The production benchmark manifest",
        "not replacements for the production-resolution gates",
    )

    for fragment in stale_fragments:
        assert fragment not in readme


def test_public_docs_are_standalone_not_development_log() -> None:
    for path in PUBLIC_STANDALONE_DOCS:
        text = path.read_text()
        checked_text = "\n".join(
            line
            for line in text.splitlines()
            if "docs.jax.dev/en/latest" not in line
        )
        for fragment in PUBLIC_STALE_FRAGMENTS:
            assert fragment not in checked_text, f"{fragment!r} appears in {path}"
        for pattern in PUBLIC_STALE_PATTERNS:
            assert not pattern.search(checked_text), f"{pattern.pattern!r} appears in {path}"


def test_rejected_benchmark_history_fragments_are_not_in_public_docs_tree() -> None:
    docs_to_scan = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "sfincs_jax" / "README.md",
        REPO_ROOT / "examples" / "README.md",
    ]
    docs_to_scan.extend(
        path
        for path in (REPO_ROOT / "docs").rglob("*")
        if path.suffix in {".md", ".rst"}
        and path.name != "release_notes.rst"
        and "upstream" not in path.parts
    )

    for path in docs_to_scan:
        text = path.read_text(encoding="utf-8")
        for fragment in DOC_TREE_STALE_FRAGMENTS:
            assert fragment not in text, f"{fragment!r} appears in {path}"


def test_testing_docs_coverage_gate_matches_ci_workflow() -> None:
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    match = re.search(r"coverage report --fail-under=(\d+)", workflow)
    assert match is not None
    fail_under = match.group(1)

    testing_docs = (REPO_ROOT / "docs" / "testing.rst").read_text(encoding="utf-8")
    assert f"CI fail-under gate is ``{fail_under}%``" in testing_docs
    assert f"``{fail_under} -> 85 -> 90 -> 95``" in testing_docs
