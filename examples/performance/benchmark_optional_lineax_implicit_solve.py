"""Benchmark the optional Lineax implicit-solve gate.

This script is deliberately outside the production solver path. It compares the
current in-tree implicit linear solve against an optional ``lineax`` solve on a
small deterministic nonsymmetric system. The gate is useful for deciding whether
Lineax is worth evaluating on real SFINCS operators; it is not a production
dependency and exits cleanly when Lineax is not installed.

Example
-------

.. code-block:: bash

   python examples/performance/benchmark_optional_lineax_implicit_solve.py \
     --backend all \
     --out-json examples/performance/output/lineax_implicit_gate.json
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

from sfincs_jax.implicit_solve import linear_custom_solve


_AUTO_IMPORT = object()


@dataclass(frozen=True)
class GateResult:
    backend: str
    status: str
    size: int
    residual_norm: float | None
    relative_residual: float | None
    objective: float | None
    grad: float | None
    finite_difference_grad: float | None
    grad_abs_error: float | None
    elapsed_s: float | None
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_nonsymmetric_system(size: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a deterministic, well-conditioned, nonsymmetric linear system."""
    n = int(size)
    if n < 2:
        raise ValueError("size must be at least 2")
    i = jnp.arange(n, dtype=jnp.float64)
    rows = i[:, None]
    cols = i[None, :]
    diag = 3.0 + 0.2 * i
    upper_distance = jnp.maximum(1.0, cols - rows)
    lower_distance = jnp.maximum(1.0, rows - cols)
    upper = jnp.where(cols > rows, 0.08 / (1.0 + upper_distance), 0.0)
    lower = jnp.where(rows > cols, -0.05 / (1.0 + lower_distance), 0.0)
    rank_one = 0.015 * jnp.sin((rows + 1.0) * (cols + 2.0))
    matrix = jnp.diag(diag) + upper + lower + rank_one
    rhs = 0.5 + jnp.cos(0.3 + i)
    return matrix.astype(jnp.float64), rhs.astype(jnp.float64)


def _finite_difference_grad(objective, p0: float, eps: float = 1.0e-5) -> float:
    f_plus = float(objective(jnp.asarray(p0 + eps, dtype=jnp.float64)))
    f_minus = float(objective(jnp.asarray(p0 - eps, dtype=jnp.float64)))
    return (f_plus - f_minus) / (2.0 * eps)


def _result_from_solution(
    *,
    backend: str,
    size: int,
    matrix: jnp.ndarray,
    rhs: jnp.ndarray,
    p0: float,
    objective,
    elapsed_s: float,
) -> GateResult:
    value, grad = jax.value_and_grad(objective)(jnp.asarray(p0, dtype=jnp.float64))
    value.block_until_ready()
    grad.block_until_ready()
    fd = _finite_difference_grad(objective, p0)
    shifted = matrix + p0 * jnp.eye(int(matrix.shape[0]), dtype=matrix.dtype)
    x = _solve_current(shifted, rhs, tol=1.0e-12, restart=min(20, int(matrix.shape[0])), maxiter=100)
    residual = rhs - shifted @ x
    residual_norm = float(jnp.linalg.norm(residual))
    rhs_norm = max(float(jnp.linalg.norm(rhs)), 1.0)
    return GateResult(
        backend=backend,
        status="ok",
        size=int(size),
        residual_norm=residual_norm,
        relative_residual=residual_norm / rhs_norm,
        objective=float(value),
        grad=float(grad),
        finite_difference_grad=float(fd),
        grad_abs_error=abs(float(grad) - float(fd)),
        elapsed_s=float(elapsed_s),
    )


def _solve_current(
    matrix: jnp.ndarray,
    rhs: jnp.ndarray,
    *,
    tol: float,
    restart: int,
    maxiter: int,
) -> jnp.ndarray:
    def matvec(x: jnp.ndarray) -> jnp.ndarray:
        return matrix @ x

    return linear_custom_solve(
        matvec=matvec,
        b=rhs,
        tol=tol,
        atol=0.0,
        restart=restart,
        maxiter=maxiter,
        solver="gmres",
        solve_method="incremental",
        solver_jit=False,
    ).x


def run_current_gate(
    *,
    matrix: jnp.ndarray,
    rhs: jnp.ndarray,
    p0: float,
    tol: float,
    restart: int,
    maxiter: int,
) -> GateResult:
    def objective(p: jnp.ndarray) -> jnp.ndarray:
        shifted = matrix + p * jnp.eye(int(matrix.shape[0]), dtype=matrix.dtype)
        x = _solve_current(shifted, rhs, tol=tol, restart=restart, maxiter=maxiter)
        return 0.5 * jnp.vdot(x, x)

    t0 = time.perf_counter()
    value, grad = jax.value_and_grad(objective)(jnp.asarray(p0, dtype=jnp.float64))
    value.block_until_ready()
    grad.block_until_ready()
    elapsed_s = time.perf_counter() - t0
    return _result_from_solution(
        backend="current_custom_linear_solve",
        size=int(matrix.shape[0]),
        matrix=matrix,
        rhs=rhs,
        p0=p0,
        objective=objective,
        elapsed_s=elapsed_s,
    )


def _import_lineax():
    try:
        import lineax as lx  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return None, exc
    return lx, None


def run_lineax_gate(
    *,
    matrix: jnp.ndarray,
    rhs: jnp.ndarray,
    p0: float,
    tol: float,
    restart: int,
    maxiter: int,
    lineax_module: Any = _AUTO_IMPORT,
) -> GateResult:
    if lineax_module is _AUTO_IMPORT:
        lx, import_error = _import_lineax()
    elif lineax_module is None:
        lx, import_error = None, ImportError("lineax was not provided")
    else:
        lx, import_error = lineax_module, None
    if lx is None:
        return GateResult(
            backend="lineax_gmres",
            status="skipped",
            size=int(matrix.shape[0]),
            residual_norm=None,
            relative_residual=None,
            objective=None,
            grad=None,
            finite_difference_grad=None,
            grad_abs_error=None,
            elapsed_s=None,
            error=f"Lineax unavailable: {import_error}",
        )

    def _solve_lineax(shifted: jnp.ndarray) -> jnp.ndarray:
        operator = lx.MatrixLinearOperator(shifted)
        solver = lx.GMRES(rtol=tol, atol=0.0, restart=restart, max_steps=maxiter)
        solution = lx.linear_solve(operator, rhs, solver=solver, throw=False)
        return solution.value

    def objective(p: jnp.ndarray) -> jnp.ndarray:
        shifted = matrix + p * jnp.eye(int(matrix.shape[0]), dtype=matrix.dtype)
        x = _solve_lineax(shifted)
        return 0.5 * jnp.vdot(x, x)

    try:
        t0 = time.perf_counter()
        value, grad = jax.value_and_grad(objective)(jnp.asarray(p0, dtype=jnp.float64))
        value.block_until_ready()
        grad.block_until_ready()
        elapsed_s = time.perf_counter() - t0
        fd = _finite_difference_grad(objective, p0)
        shifted = matrix + p0 * jnp.eye(int(matrix.shape[0]), dtype=matrix.dtype)
        x = _solve_lineax(shifted)
        residual = rhs - shifted @ x
        residual_norm = float(jnp.linalg.norm(residual))
        rhs_norm = max(float(jnp.linalg.norm(rhs)), 1.0)
        return GateResult(
            backend="lineax_gmres",
            status="ok",
            size=int(matrix.shape[0]),
            residual_norm=residual_norm,
            relative_residual=residual_norm / rhs_norm,
            objective=float(value),
            grad=float(grad),
            finite_difference_grad=float(fd),
            grad_abs_error=abs(float(grad) - float(fd)),
            elapsed_s=float(elapsed_s),
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            backend="lineax_gmres",
            status="error",
            size=int(matrix.shape[0]),
            residual_norm=None,
            relative_residual=None,
            objective=None,
            grad=None,
            finite_difference_grad=None,
            grad_abs_error=None,
            elapsed_s=None,
            error=str(exc),
        )


def run_gate(args: argparse.Namespace) -> list[GateResult]:
    matrix, rhs = make_nonsymmetric_system(int(args.size))
    restart = min(int(args.restart), int(args.size))
    backends = ["current", "lineax"] if args.backend == "all" else [str(args.backend)]
    results: list[GateResult] = []
    if "current" in backends:
        results.append(
            run_current_gate(
                matrix=matrix,
                rhs=rhs,
                p0=float(args.shift),
                tol=float(args.tol),
                restart=restart,
                maxiter=int(args.maxiter),
            )
        )
    if "lineax" in backends:
        results.append(
            run_lineax_gate(
                matrix=matrix,
                rhs=rhs,
                p0=float(args.shift),
                tol=float(args.tol),
                restart=restart,
                maxiter=int(args.maxiter),
            )
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("current", "lineax", "all"), default="all")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--shift", type=float, default=0.2)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--restart", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--out-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    results = run_gate(args)
    payload = [result.to_json_dict() for result in results]
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
