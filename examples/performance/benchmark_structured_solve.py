"""Benchmark factor-once structured solves against dense repeated solves.

This harness is intentionally synthetic and bounded. It exercises the block-
tridiagonal solver used for structured velocity-space experiments without
changing production defaults. Use it as an admission gate before porting
factor-once / repeated-RHS ideas into real SFINCS operators.

Example
-------

.. code-block:: bash

   python examples/performance/benchmark_structured_solve.py \
     --nblocks 32 --block-size 8 --n-rhs 8 \
     --out-json examples/performance/output/structured_solve_gate.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.structured_velocity import (  # noqa: E402
    block_tridiagonal_to_dense,
    factor_block_tridiagonal,
)


@dataclass(frozen=True)
class StructuredSolveResult:
    status: str
    platform: str
    nblocks: int
    block_size: int
    n_rhs: int
    size: int
    dense_bytes: int
    structured_bytes: int
    dense_solve_s: float
    structured_factor_s: float
    structured_solve_s: float
    structured_total_s: float
    speedup_vs_dense_solve: float
    dense_relative_residual: float
    structured_relative_residual: float
    max_solution_error: float

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_structured_problem(
    *, nblocks: int, block_size: int, n_rhs: int, seed: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build a deterministic well-conditioned block-tridiagonal test problem."""
    if int(nblocks) < 2:
        raise ValueError("nblocks must be at least 2")
    if int(block_size) < 1:
        raise ValueError("block_size must be at least 1")
    if int(n_rhs) < 1:
        raise ValueError("n_rhs must be at least 1")

    rng = np.random.default_rng(int(seed))
    nblocks_i = int(nblocks)
    block_size_i = int(block_size)
    n_rhs_i = int(n_rhs)

    eye = np.eye(block_size_i, dtype=np.float64)
    diagonal = np.empty((nblocks_i, block_size_i, block_size_i), dtype=np.float64)
    for k in range(nblocks_i):
        symmetric_noise = rng.standard_normal((block_size_i, block_size_i))
        symmetric_noise = 0.5 * (symmetric_noise + symmetric_noise.T)
        diagonal[k] = (4.0 + 0.05 * k) * eye + 0.02 * symmetric_noise
    lower = 0.04 * rng.standard_normal((nblocks_i - 1, block_size_i, block_size_i))
    upper = 0.04 * rng.standard_normal((nblocks_i - 1, block_size_i, block_size_i))
    rhs = rng.standard_normal((n_rhs_i, nblocks_i * block_size_i)).astype(np.float64)
    return (
        jnp.asarray(diagonal, dtype=jnp.float64),
        jnp.asarray(lower, dtype=jnp.float64),
        jnp.asarray(upper, dtype=jnp.float64),
        jnp.asarray(rhs, dtype=jnp.float64),
    )


def _time_call(fn, *, warmup: int, repeats: int) -> tuple[Any, float]:
    result = None
    for _ in range(max(0, int(warmup))):
        result = fn()
        jax.block_until_ready(result)
    elapsed: list[float] = []
    for _ in range(max(1, int(repeats))):
        t0 = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        elapsed.append(time.perf_counter() - t0)
    return result, float(min(elapsed))


def _relative_residual(matrix: jnp.ndarray, x: jnp.ndarray, rhs: jnp.ndarray) -> float:
    residual = rhs - jnp.einsum("ij,bj->bi", matrix, x)
    denom = jnp.maximum(jnp.linalg.norm(rhs), 1.0)
    return float(jnp.linalg.norm(residual) / denom)


def run_benchmark(
    *,
    nblocks: int = 16,
    block_size: int = 8,
    n_rhs: int = 6,
    seed: int = 1,
    warmup: int = 1,
    repeats: int = 3,
) -> StructuredSolveResult:
    diagonal, lower, upper, rhs = make_structured_problem(
        nblocks=nblocks,
        block_size=block_size,
        n_rhs=n_rhs,
        seed=seed,
    )
    size = int(nblocks) * int(block_size)
    dense = block_tridiagonal_to_dense(diagonal, lower, upper)

    dense_solve = jax.jit(lambda a, b: jnp.linalg.solve(a, b.T).T)
    factor_once = jax.jit(lambda d, lower_blocks, u: factor_block_tridiagonal(d, lower_blocks, u))
    solve_structured = jax.jit(lambda factor, b: factor.solve(b))

    dense_x, dense_solve_s = _time_call(
        lambda: dense_solve(dense, rhs),
        warmup=warmup,
        repeats=repeats,
    )
    factor, structured_factor_s = _time_call(
        lambda: factor_once(diagonal, lower, upper),
        warmup=warmup,
        repeats=repeats,
    )
    structured_x, structured_solve_s = _time_call(
        lambda: solve_structured(factor, rhs),
        warmup=warmup,
        repeats=repeats,
    )

    dense_relative_residual = _relative_residual(dense, dense_x, rhs)
    structured_relative_residual = _relative_residual(dense, structured_x, rhs)
    max_solution_error = float(jnp.max(jnp.abs(structured_x - dense_x)))

    dense_bytes = int(size * size * np.dtype(np.float64).itemsize)
    structured_block_count = int(nblocks) + 2 * max(0, int(nblocks) - 1)
    structured_bytes = int(structured_block_count * int(block_size) * int(block_size) * np.dtype(np.float64).itemsize)
    structured_total_s = float(structured_factor_s + structured_solve_s)
    speedup = float(dense_solve_s / structured_total_s) if structured_total_s > 0.0 else float("inf")

    return StructuredSolveResult(
        status="ok",
        platform=str(jax.default_backend()),
        nblocks=int(nblocks),
        block_size=int(block_size),
        n_rhs=int(n_rhs),
        size=int(size),
        dense_bytes=int(dense_bytes),
        structured_bytes=int(structured_bytes),
        dense_solve_s=float(dense_solve_s),
        structured_factor_s=float(structured_factor_s),
        structured_solve_s=float(structured_solve_s),
        structured_total_s=float(structured_total_s),
        speedup_vs_dense_solve=float(speedup),
        dense_relative_residual=float(dense_relative_residual),
        structured_relative_residual=float(structured_relative_residual),
        max_solution_error=float(max_solution_error),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nblocks", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--n-rhs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=_REPO_ROOT / "examples" / "performance" / "output" / "structured_solve_gate.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_benchmark(
        nblocks=int(args.nblocks),
        block_size=int(args.block_size),
        n_rhs=int(args.n_rhs),
        seed=int(args.seed),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result.to_json_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
