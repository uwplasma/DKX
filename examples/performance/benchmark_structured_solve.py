"""Benchmark factor-once structured solves against dense repeated solves.

This harness is intentionally bounded. It exercises the block-tridiagonal solver
used for structured velocity-space experiments without changing production
defaults. Use it as an admission gate before porting factor-once / repeated-RHS
ideas into real SFINCS operators.

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
from sfincs_jax.namelist import read_sfincs_input  # noqa: E402
from sfincs_jax.v3_fblock import apply_v3_fblock_operator  # noqa: E402
from sfincs_jax.v3_system import full_system_operator_from_namelist  # noqa: E402


_DEFAULT_SFINCS_BLOCK_INPUT = _REPO_ROOT / "tests" / "ref" / "monoenergetic_PAS_tiny_scheme1.input.namelist"


@dataclass(frozen=True)
class StructuredSolveResult:
    status: str
    case: str
    platform: str
    source_input: str | None
    nblocks: int
    block_size: int
    n_rhs: int
    size: int
    dense_bytes: int
    structured_bytes: int
    off_band_norm: float
    regularization: float
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


def extract_sfincs_pas_block_problem(
    *,
    input_path: Path,
    species_index: int = 0,
    x_index: int = 0,
    n_rhs: int = 4,
    seed: int = 1,
    off_band_tol: float = 1.0e-10,
    regularization: float = 1.0e-4,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    """Extract one real PAS local block from the SFINCS F-block operator.

    The block fixes species and speed index, then retains all active Legendre
    modes and all angular grid points. For PAS/no-Er monoenergetic fixtures this
    is block-tridiagonal in Legendre index, with each block spanning
    ``(theta,zeta)``.
    """
    nml = read_sfincs_input(Path(input_path))
    op = full_system_operator_from_namelist(nml=nml).fblock
    if op.pas is None:
        raise ValueError(f"{input_path} is not a PAS fixture")
    if op.fp is not None:
        raise ValueError(f"{input_path} includes a Fokker-Planck operator; expected PAS-only")

    species = int(species_index)
    x_idx = int(x_index)
    if not (0 <= species < int(op.n_species)):
        raise ValueError(f"species_index={species} outside [0, {int(op.n_species)})")
    if not (0 <= x_idx < int(op.n_x)):
        raise ValueError(f"x_index={x_idx} outside [0, {int(op.n_x)})")

    n_xi_for_x = np.asarray(op.collisionless.n_xi_for_x, dtype=np.int32)
    n_l = int(n_xi_for_x[x_idx])
    if n_l < 2:
        raise ValueError(f"selected x_index={x_idx} has only {n_l} active Legendre modes")
    block_size = int(op.n_theta * op.n_zeta)
    size = int(n_l * block_size)

    columns: list[np.ndarray] = []
    for col in range(size):
        ell = int(col // block_size)
        rem = int(col % block_size)
        theta = int(rem // int(op.n_zeta))
        zeta = int(rem % int(op.n_zeta))
        basis = jnp.zeros(op.f_shape, dtype=jnp.float64).at[species, x_idx, ell, theta, zeta].set(1.0)
        applied = apply_v3_fblock_operator(op, basis)
        columns.append(np.asarray(applied[species, x_idx, :n_l, :, :], dtype=np.float64).reshape(-1))
    dense = np.stack(columns, axis=1)

    diagonal = np.empty((n_l, block_size, block_size), dtype=np.float64)
    lower = np.empty((n_l - 1, block_size, block_size), dtype=np.float64)
    upper = np.empty((n_l - 1, block_size, block_size), dtype=np.float64)
    off_band_norm = 0.0
    for row_block in range(n_l):
        row = slice(row_block * block_size, (row_block + 1) * block_size)
        diagonal[row_block] = dense[row, row]
        for col_block in range(n_l):
            col = slice(col_block * block_size, (col_block + 1) * block_size)
            block = dense[row, col]
            if row_block == col_block + 1:
                lower[col_block] = block
            elif row_block + 1 == col_block:
                upper[row_block] = block
            elif abs(row_block - col_block) > 1:
                off_band_norm = max(off_band_norm, float(np.linalg.norm(block)))
    if off_band_norm > float(off_band_tol):
        raise ValueError(
            f"selected SFINCS block is not block-tridiagonal enough: off_band_norm={off_band_norm:.3e}"
        )
    reg = max(0.0, float(regularization))
    if reg > 0.0:
        diagonal = diagonal + reg * np.eye(block_size, dtype=np.float64)[None, :, :]

    rng = np.random.default_rng(int(seed))
    rhs = rng.standard_normal((int(n_rhs), size)).astype(np.float64)
    return (
        jnp.asarray(diagonal, dtype=jnp.float64),
        jnp.asarray(lower, dtype=jnp.float64),
        jnp.asarray(upper, dtype=jnp.float64),
        jnp.asarray(rhs, dtype=jnp.float64),
        float(off_band_norm),
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
    return run_structured_problem(
        diagonal=diagonal,
        lower=lower,
        upper=upper,
        rhs=rhs,
        case="synthetic_block_tridiagonal",
        source_input=None,
        off_band_norm=0.0,
        regularization=0.0,
        warmup=warmup,
        repeats=repeats,
    )


def run_sfincs_pas_block_benchmark(
    *,
    input_path: Path = _DEFAULT_SFINCS_BLOCK_INPUT,
    species_index: int = 0,
    x_index: int = 0,
    n_rhs: int = 4,
    seed: int = 1,
    warmup: int = 1,
    repeats: int = 3,
    regularization: float = 1.0e-4,
) -> StructuredSolveResult:
    diagonal, lower, upper, rhs, off_band_norm = extract_sfincs_pas_block_problem(
        input_path=Path(input_path),
        species_index=species_index,
        x_index=x_index,
        n_rhs=n_rhs,
        seed=seed,
        regularization=regularization,
    )
    return run_structured_problem(
        diagonal=diagonal,
        lower=lower,
        upper=upper,
        rhs=rhs,
        case="sfincs_pas_local_block",
        source_input=str(Path(input_path)),
        off_band_norm=off_band_norm,
        regularization=float(regularization),
        warmup=warmup,
        repeats=repeats,
    )


def run_structured_problem(
    *,
    diagonal: jnp.ndarray,
    lower: jnp.ndarray,
    upper: jnp.ndarray,
    rhs: jnp.ndarray,
    case: str,
    source_input: str | None,
    off_band_norm: float,
    regularization: float,
    warmup: int,
    repeats: int,
) -> StructuredSolveResult:
    nblocks = int(diagonal.shape[0])
    block_size = int(diagonal.shape[1])
    n_rhs = int(rhs.shape[0])
    size = int(nblocks * block_size)
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
        case=str(case),
        platform=str(jax.default_backend()),
        source_input=source_input,
        nblocks=int(nblocks),
        block_size=int(block_size),
        n_rhs=int(n_rhs),
        size=int(size),
        dense_bytes=int(dense_bytes),
        structured_bytes=int(structured_bytes),
        off_band_norm=float(off_band_norm),
        regularization=float(regularization),
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
    parser.add_argument(
        "--case",
        choices=("synthetic", "sfincs-pas-block"),
        default="synthetic",
        help="Benchmark case. The sfincs-pas-block case extracts a real local PAS block from a SFINCS fixture.",
    )
    parser.add_argument(
        "--sfincs-input",
        type=Path,
        default=_DEFAULT_SFINCS_BLOCK_INPUT,
        help="Input namelist used when --case=sfincs-pas-block.",
    )
    parser.add_argument("--species-index", type=int, default=0)
    parser.add_argument("--x-index", type=int, default=0)
    parser.add_argument(
        "--regularization",
        type=float,
        default=1.0e-4,
        help="Diagonal regularization for extracted real SFINCS blocks.",
    )
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
    if str(args.case) == "sfincs-pas-block":
        result = run_sfincs_pas_block_benchmark(
            input_path=Path(args.sfincs_input),
            species_index=int(args.species_index),
            x_index=int(args.x_index),
            n_rhs=int(args.n_rhs),
            seed=int(args.seed),
            warmup=int(args.warmup),
            repeats=int(args.repeats),
            regularization=float(args.regularization),
        )
    else:
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
