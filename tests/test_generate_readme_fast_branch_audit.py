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
    assert table[2].startswith(
        "| `case_a` | 10.000 | 3.000 | 0.30x | 2.000 | 0.20x | "
        "4.000 | 0.40x | 1.500 | 0.15x |"
    )
