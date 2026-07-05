from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_readme_fast_branch_audit.py"
    spec = importlib.util.spec_from_file_location("generate_readme_fast_branch_audit", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_case_table_reports_cold_and_warm_logged_runtimes() -> None:
    module = _load_module()
    cpu_row = {
        "case": "case_a",
        "fortran_runtime_s": 10.0,
        "fortran_max_rss_mb": 100.0,
        "jax_runtime_s": 3.0,
        "jax_runtime_s_warm": 2.0,
        "jax_logged_elapsed_s": 2.5,
        "jax_max_rss_mb": 150.0,
        "jax_incremental_max_rss_mb": 60.0,
        "status": "parity_ok",
        "n_mismatch_common": 0,
        "n_common_keys": 12,
        "strict_n_mismatch_common": 0,
        "strict_n_common_keys": 12,
        "print_parity_signals": 3,
        "print_parity_total": 3,
    }
    gpu_row = {
        "case": "case_a",
        "fortran_runtime_s": 10.0,
        "fortran_max_rss_mb": 100.0,
        "jax_runtime_s": 4.0,
        "jax_runtime_s_warm": None,
        "jax_logged_elapsed_s": 1.5,
        "jax_max_rss_mb": 200.0,
        "jax_incremental_max_rss_mb": 75.0,
        "status": "parity_ok",
        "n_mismatch_common": 0,
        "n_common_keys": 12,
        "strict_n_mismatch_common": 0,
        "strict_n_common_keys": 12,
        "print_parity_signals": 3,
        "print_parity_total": 3,
    }

    table = module._format_case_table(
        ["case_a"],
        {"case_a": cpu_row},
        {"case_a": gpu_row},
    )

    assert "JAX CPU cold(s)" in table[0]
    assert "JAX CPU warm/logged(s)" in table[0]
    assert "JAX GPU cold(s)" in table[0]
    assert "JAX GPU warm/logged(s)" in table[0]
    assert "JAX CPU active MB" in table[0]
    assert "JAX GPU active MB" in table[0]
    assert table[2].startswith(
        "| `case_a` | 10.000 | 3.000 | 0.30x | 2.000 | 0.20x | "
        "4.000 | 0.40x | 1.500 | 0.15x |"
    )
    assert " | 100.0 | 60.0 | 0.60x | 75.0 | 0.75x | " in table[2]


def test_public_comparison_cases_filter_short_fortran_reference_runs() -> None:
    module = _load_module()
    rows_by_case = {
        "short_case": {"case": "short_case", "fortran_runtime_s": 0.7},
        "production_case": {"case": "production_case", "fortran_runtime_s": 12.0},
    }
    included, excluded = module._public_comparison_cases(
        ["short_case", "production_case"],
        rows_by_case,
        {},
        min_fortran_runtime_s=10.0,
    )

    assert included == ["production_case"]
    assert excluded == [{"case": "short_case", "fortran_runtime_s": 0.7}]
    assert "`short_case` (0.700s)" in module._format_excluded_public_cases(excluded)


def test_runtime_drift_summary_skips_mismatched_resolution_tiers() -> None:
    module = _load_module()

    lines = [
        module._format_runtime_drift_summary(
            "GPU",
            {
                "status": "not_applicable",
                "reason": reason,
                "flagged_cases": 999,
            },
        )
        for reason in (
            "production-floor reruns are not same-resolution with the frozen smoke baseline",
            "production-floor reruns are not same-resolution with the older frozen smoke baseline",
        )
    ]

    assert lines == [
        (
            "- GPU runtime drift gate: not applicable: "
            "suite rows are not same-resolution with the optional runtime baseline"
        ),
        (
            "- GPU runtime drift gate: not applicable: "
            "suite rows are not same-resolution with the optional runtime baseline"
        ),
    ]


def test_runtime_drift_summary_keeps_actionable_custom_reason() -> None:
    module = _load_module()

    line = module._format_runtime_drift_summary(
        "CPU",
        {"status": "skipped", "reason": "missing same-resolution baseline"},
    )

    assert line == (
        "- CPU runtime drift gate: not applicable: missing same-resolution baseline"
    )
