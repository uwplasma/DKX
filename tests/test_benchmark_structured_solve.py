from __future__ import annotations

import json
from pathlib import Path

from examples.performance.benchmark_structured_solve import (
    main,
    make_structured_problem,
    run_benchmark,
)


def test_make_structured_problem_shapes_are_deterministic() -> None:
    diagonal, lower, upper, rhs = make_structured_problem(
        nblocks=5,
        block_size=3,
        n_rhs=2,
        seed=7,
    )

    assert diagonal.shape == (5, 3, 3)
    assert lower.shape == (4, 3, 3)
    assert upper.shape == (4, 3, 3)
    assert rhs.shape == (2, 15)


def test_structured_solve_benchmark_is_accuracy_and_memory_gate() -> None:
    result = run_benchmark(
        nblocks=5,
        block_size=3,
        n_rhs=2,
        seed=8,
        warmup=0,
        repeats=1,
    )

    assert result.status == "ok"
    assert result.size == 15
    assert result.structured_bytes < result.dense_bytes
    assert result.structured_relative_residual < 1.0e-11
    assert result.dense_relative_residual < 1.0e-11
    assert result.max_solution_error < 1.0e-11
    assert result.structured_total_s >= 0.0
    assert result.dense_solve_s >= 0.0


def test_structured_solve_benchmark_main_writes_json(tmp_path: Path) -> None:
    out_json = tmp_path / "structured.json"

    assert main(
        [
            "--nblocks",
            "4",
            "--block-size",
            "2",
            "--n-rhs",
            "2",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--out-json",
            str(out_json),
        ]
    ) == 0

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["size"] == 8
    assert payload["structured_bytes"] < payload["dense_bytes"]
