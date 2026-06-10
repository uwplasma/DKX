from __future__ import annotations

import argparse
from pathlib import Path

from scripts.benchmark_rhs1_fblock_csr_reuse import DEFAULT_INPUT, run_benchmark


def test_rhs1_fblock_csr_reuse_benchmark_dry_run() -> None:
    payload = run_benchmark(
        argparse.Namespace(
            input=DEFAULT_INPUT,
            out=Path("/tmp/unused.json"),
            identity_shift=0.5,
            repeats=2,
            max_csr_mb=128.0,
            dry_run=True,
            json=False,
        )
    )

    assert payload["kind"] == "rhs1_fblock_csr_reuse_benchmark"
    assert payload["status"] == "planned"
    assert payload["dry_run"] is True


def test_rhs1_fblock_csr_reuse_benchmark_real_tiny_case() -> None:
    payload = run_benchmark(
        argparse.Namespace(
            input=DEFAULT_INPUT,
            out=Path("/tmp/unused.json"),
            identity_shift=0.5,
            repeats=2,
            max_csr_mb=128.0,
            dry_run=False,
            json=False,
        )
    )

    assert payload["status"] == "ok"
    assert payload["cache_hits"] == 1
    assert payload["max_abs_error"] < 1.0e-12
    assert payload["max_rel_error"] < 1.0e-12
    assert payload["rows"][0]["cache_hit"] is False
    assert payload["rows"][1]["cache_hit"] is True
