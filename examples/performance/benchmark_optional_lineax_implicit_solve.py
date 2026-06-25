"""Benchmark the optional Lineax implicit-solve gate.

This script is deliberately outside the production solver path. It compares the
current in-tree implicit linear solve against an optional ``lineax`` solve on:

- a small deterministic nonsymmetric system,
- a tiny real SFINCS implicit-diff operator,
- and a repeated-RHS reuse case on that same tiny real operator.

The gate is useful for deciding whether Lineax is worth evaluating on real
SFINCS operators; it is not a production dependency and exits cleanly when
Lineax is not installed.

Example
-------

.. code-block:: bash

   python examples/performance/benchmark_optional_lineax_implicit_solve.py \
     --backend all \
     --suite all \
     --out-json examples/performance/output/lineax_implicit_gate.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.solvers.implicit import linear_custom_solve  # noqa: E402
from sfincs_jax.namelist import read_sfincs_input  # noqa: E402
from sfincs_jax.petsc_binary import read_petsc_vec  # noqa: E402
from sfincs_jax.v3_system import (  # noqa: E402
    apply_v3_full_system_operator,
    full_system_operator_from_namelist,
    rhs_v3_full_system,
)


_AUTO_IMPORT = object()


@dataclass(frozen=True)
class GateResult:
    case: str
    backend: str
    status: str
    size: int
    n_rhs: int
    residual_norm: float | None
    relative_residual: float | None
    max_solution_error: float | None
    objective: float | None
    grad: float | None
    finite_difference_grad: float | None
    grad_abs_error: float | None
    elapsed_s: float | None
    error: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline <= 0.0:
        return None
    return float(candidate) / float(baseline)


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None


def _lineax_numeric_gate_passed(row: dict[str, Any]) -> bool:
    """Return True for rows that are numerically clean despite solver status."""
    relative_residual = _finite_float(row.get("relative_residual"))
    if relative_residual is None or relative_residual >= 1.0e-8:
        return False
    grad_abs_error = _finite_float(row.get("grad_abs_error"))
    if grad_abs_error is not None and grad_abs_error >= 1.0e-4:
        return False
    max_solution_error = _finite_float(row.get("max_solution_error"))
    if max_solution_error is not None and max_solution_error >= 1.0e-6:
        return False
    return True


def summarize_gate_results(results: list[GateResult]) -> dict[str, Any]:
    """Summarize measured Lineax evidence into a bounded adoption decision."""
    rows = [result.to_json_dict() for result in results]
    counts = Counter(str(row["status"]) for row in rows)
    measured_rows = [
        row for row in rows if row.get("status") == "ok" and row.get("elapsed_s") is not None
    ]
    by_case_backend = {(str(row["case"]), str(row["backend"])): row for row in rows}
    comparisons: list[dict[str, Any]] = []
    for case in sorted({str(row["case"]) for row in rows}):
        current = by_case_backend.get((case, "current_custom_linear_solve"))
        lineax = by_case_backend.get((case, "lineax_gmres"))
        if current is None or lineax is None:
            continue
        comparisons.append(
            {
                "case": case,
                "current_status": current["status"],
                "lineax_status": lineax["status"],
                "elapsed_ratio_lineax_over_current": _safe_ratio(
                    lineax.get("elapsed_s"),
                    current.get("elapsed_s"),
                ),
                "current_relative_residual": current.get("relative_residual"),
                "lineax_relative_residual": lineax.get("relative_residual"),
                "lineax_grad_abs_error": lineax.get("grad_abs_error"),
                "lineax_max_solution_error": lineax.get("max_solution_error"),
            }
        )

    lineax_rows = [row for row in rows if row["backend"] == "lineax_gmres"]
    lineax_ok = [row for row in lineax_rows if row["status"] == "ok"]
    lineax_errors = [row for row in lineax_rows if row["status"] == "error"]
    lineax_skipped = [row for row in lineax_rows if row["status"] == "skipped"]
    lineax_status_mismatches = [
        row for row in lineax_errors if _lineax_numeric_gate_passed(row)
    ]
    sfincs_lineax_status_mismatches = [
        row for row in lineax_status_mismatches if str(row["case"]).startswith("sfincs_")
    ]
    sfincs_lineax_rows = [row for row in lineax_rows if str(row["case"]).startswith("sfincs_")]
    sfincs_lineax_ready = bool(sfincs_lineax_rows) and all(
        row["status"] == "ok"
        and (row.get("relative_residual") is None or float(row["relative_residual"]) < 1.0e-8)
        and (row.get("grad_abs_error") is None or float(row["grad_abs_error"]) < 1.0e-4)
        and (row.get("max_solution_error") is None or float(row["max_solution_error"]) < 1.0e-6)
        for row in sfincs_lineax_rows
    )
    speedup_cases = [
        row for row in comparisons
        if row["elapsed_ratio_lineax_over_current"] is not None
        and row["lineax_status"] == "ok"
        and float(row["elapsed_ratio_lineax_over_current"]) < 0.9
    ]

    if sfincs_lineax_status_mismatches:
        decision = "do_not_promote_lineax_status_mismatch"
        reason = (
            "Lineax residuals were numerically clean on real SFINCS rows, "
            "but solver result statuses were not successful."
        )
    elif lineax_errors:
        decision = "do_not_promote_error_status"
        reason = "At least one Lineax row returned an error status."
    elif lineax_skipped and not lineax_ok:
        decision = "not_evaluated_missing_optional_dependency"
        reason = "Lineax is not installed in this environment."
    elif sfincs_lineax_ready and speedup_cases:
        decision = "candidate_for_bounded_experiments_only"
        reason = "Lineax passed SFINCS residual checks and showed at least one measured speedup."
    elif lineax_ok:
        decision = "defer_no_sufficient_sfincs_speedup_evidence"
        reason = "Lineax ran, but the gate does not yet justify production adoption."
    else:
        decision = "not_evaluated"
        reason = "No Lineax evidence rows were available."

    return {
        "gate": "optional_lineax_implicit_solve",
        "rows": len(rows),
        "status_counts": dict(counts),
        "measured_rows": len(measured_rows),
        "backends": sorted({str(row["backend"]) for row in rows}),
        "cases": sorted({str(row["case"]) for row in rows}),
        "comparisons": comparisons,
        "lineax_evidence": {
            "ok_rows": len(lineax_ok),
            "skipped_rows": len(lineax_skipped),
            "error_rows": len(lineax_errors),
            "residual_clean_status_mismatch_rows": len(lineax_status_mismatches),
            "sfincs_residual_clean_status_mismatch_cases": [
                str(row["case"]) for row in sfincs_lineax_status_mismatches
            ],
            "sfincs_rows_ready": sfincs_lineax_ready,
            "speedup_cases": [row["case"] for row in speedup_cases],
        },
        "adoption_decision": {
            "lineax": decision,
            "reason": reason,
            "production_default": "keep_current_custom_linear_solve",
            "hard_dependency": False,
        },
    }


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


def _default_sfincs_input() -> Path:
    return _REPO_ROOT / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"


def _statevector_path(input_path: Path) -> Path:
    return Path(str(input_path).replace(".input.namelist", ".stateVector.petscbin"))


@lru_cache(maxsize=4)
def load_tiny_sfincs_fixture(input_path_str: str) -> tuple[object, jnp.ndarray, float]:
    """Load the tiny scheme-5 PAS full-system operator and reference state."""
    input_path = Path(input_path_str)
    nml = read_sfincs_input(input_path)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    x_ref = jnp.asarray(read_petsc_vec(_statevector_path(input_path)).values, dtype=jnp.float64)
    if op0.fblock.pas is None:
        raise ValueError(f"{input_path} is expected to be a PAS fixture with differentiable nu_n.")
    nu0 = float(op0.fblock.pas.nu_n)
    return op0, x_ref, nu0


def _lineax_result_name(lx, result: Any) -> str:
    try:
        return str(lx.RESULTS[result])
    except Exception:  # noqa: BLE001
        return str(result)


def _sfincs_gate_solver_window(restart: int, maxiter: int) -> tuple[int, int]:
    """Use a parity-clean Krylov window for the tiny real SFINCS gate."""
    return max(80, int(restart)), max(400, int(maxiter))


def _finite_difference_grad(objective, p0: float, eps: float = 1.0e-5) -> float:
    f_plus = float(objective(jnp.asarray(p0 + eps, dtype=jnp.float64)))
    f_minus = float(objective(jnp.asarray(p0 - eps, dtype=jnp.float64)))
    return (f_plus - f_minus) / (2.0 * eps)


def _result_from_solution(
    *,
    case: str,
    backend: str,
    size: int,
    n_rhs: int,
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
        case=case,
        backend=backend,
        status="ok",
        size=int(size),
        n_rhs=int(n_rhs),
        residual_norm=residual_norm,
        relative_residual=residual_norm / rhs_norm,
        max_solution_error=None,
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
    case: str,
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
        case=case,
        backend="current_custom_linear_solve",
        size=int(matrix.shape[0]),
        n_rhs=1,
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
    case: str,
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
            case=case,
            backend="lineax_gmres",
            status="skipped",
            size=int(matrix.shape[0]),
            n_rhs=1,
            residual_norm=None,
            relative_residual=None,
            max_solution_error=None,
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
            case=case,
            backend="lineax_gmres",
            status="ok",
            size=int(matrix.shape[0]),
            n_rhs=1,
            residual_norm=residual_norm,
            relative_residual=residual_norm / rhs_norm,
            max_solution_error=None,
            objective=float(value),
            grad=float(grad),
            finite_difference_grad=float(fd),
            grad_abs_error=abs(float(grad) - float(fd)),
            elapsed_s=float(elapsed_s),
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            case=case,
            backend="lineax_gmres",
            status="error",
            size=int(matrix.shape[0]),
            n_rhs=1,
            residual_norm=None,
            relative_residual=None,
            max_solution_error=None,
            objective=None,
            grad=None,
            finite_difference_grad=None,
            grad_abs_error=None,
            elapsed_s=None,
            error=str(exc),
        )


def run_current_sfincs_implicit_gate(
    *,
    input_path: Path,
    tol: float,
    restart: int,
    maxiter: int,
) -> GateResult:
    op0, _x_ref, nu0 = load_tiny_sfincs_fixture(str(input_path))
    restart_use, maxiter_use = _sfincs_gate_solver_window(restart, maxiter)

    def objective(nu_n: jnp.ndarray) -> jnp.ndarray:
        pas2 = replace(op0.fblock.pas, nu_n=jnp.asarray(nu_n, dtype=jnp.float64))
        op = replace(op0, fblock=replace(op0.fblock, pas=pas2))
        rhs = rhs_v3_full_system(op)

        def mv(x: jnp.ndarray) -> jnp.ndarray:
            return apply_v3_full_system_operator(op, x)

        x = linear_custom_solve(
            matvec=mv,
            b=rhs,
            tol=tol,
            atol=0.0,
            restart=restart_use,
            maxiter=maxiter_use,
            solver="gmres",
            solve_method="incremental",
            solver_jit=False,
        ).x
        return 0.5 * jnp.vdot(x, x)

    t0 = time.perf_counter()
    value, grad = jax.value_and_grad(objective)(jnp.asarray(nu0, dtype=jnp.float64))
    value.block_until_ready()
    grad.block_until_ready()
    elapsed_s = time.perf_counter() - t0

    pas2 = replace(op0.fblock.pas, nu_n=jnp.asarray(nu0, dtype=jnp.float64))
    op = replace(op0, fblock=replace(op0.fblock, pas=pas2))
    rhs = rhs_v3_full_system(op)
    x = linear_custom_solve(
        matvec=lambda v: apply_v3_full_system_operator(op, v),
        b=rhs,
        tol=tol,
        atol=0.0,
        restart=restart_use,
        maxiter=maxiter_use,
        solver="gmres",
        solve_method="incremental",
        solver_jit=False,
    ).x
    residual = rhs - apply_v3_full_system_operator(op, x)
    residual_norm = float(jnp.linalg.norm(residual))
    rhs_norm = max(float(jnp.linalg.norm(rhs)), 1.0)
    fd = _finite_difference_grad(objective, nu0)
    return GateResult(
        case="sfincs_tiny_implicit",
        backend="current_custom_linear_solve",
        status="ok",
        size=int(op.total_size),
        n_rhs=1,
        residual_norm=residual_norm,
        relative_residual=residual_norm / rhs_norm,
        max_solution_error=None,
        objective=float(value),
        grad=float(grad),
        finite_difference_grad=float(fd),
        grad_abs_error=abs(float(grad) - float(fd)),
        elapsed_s=float(elapsed_s),
    )


def run_lineax_sfincs_implicit_gate(
    *,
    input_path: Path,
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
    op0, _x_ref, nu0 = load_tiny_sfincs_fixture(str(input_path))
    restart_use, maxiter_use = _sfincs_gate_solver_window(restart, maxiter)
    if lx is None:
        return GateResult(
            case="sfincs_tiny_implicit",
            backend="lineax_gmres",
            status="skipped",
            size=int(op0.total_size),
            n_rhs=1,
            residual_norm=None,
            relative_residual=None,
            max_solution_error=None,
            objective=None,
            grad=None,
            finite_difference_grad=None,
            grad_abs_error=None,
            elapsed_s=None,
            error=f"Lineax unavailable: {import_error}",
        )

    def _solve_lineax(op, rhs: jnp.ndarray) -> tuple[jnp.ndarray, str]:
        input_structure = jax.ShapeDtypeStruct(rhs.shape, rhs.dtype)
        operator = lx.FunctionLinearOperator(lambda v: apply_v3_full_system_operator(op, v), input_structure)
        solver = lx.GMRES(rtol=tol, atol=0.0, restart=restart_use, max_steps=maxiter_use)
        solution = lx.linear_solve(operator, rhs, solver=solver, throw=False)
        return solution.value, _lineax_result_name(lx, solution.result)

    def objective(nu_n: jnp.ndarray) -> jnp.ndarray:
        pas2 = replace(op0.fblock.pas, nu_n=jnp.asarray(nu_n, dtype=jnp.float64))
        op = replace(op0, fblock=replace(op0.fblock, pas=pas2))
        rhs = rhs_v3_full_system(op)
        x, _result_name = _solve_lineax(op, rhs)
        return 0.5 * jnp.vdot(x, x)

    try:
        t0 = time.perf_counter()
        value, grad = jax.value_and_grad(objective)(jnp.asarray(nu0, dtype=jnp.float64))
        value.block_until_ready()
        grad.block_until_ready()
        elapsed_s = time.perf_counter() - t0
        pas2 = replace(op0.fblock.pas, nu_n=jnp.asarray(nu0, dtype=jnp.float64))
        op = replace(op0, fblock=replace(op0.fblock, pas=pas2))
        rhs = rhs_v3_full_system(op)
        x, result_name = _solve_lineax(op, rhs)
        residual = rhs - apply_v3_full_system_operator(op, x)
        residual_norm = float(jnp.linalg.norm(residual))
        rhs_norm = max(float(jnp.linalg.norm(rhs)), 1.0)
        fd = _finite_difference_grad(objective, nu0)
        status = "ok" if result_name == "successful" else "error"
        return GateResult(
            case="sfincs_tiny_implicit",
            backend="lineax_gmres",
            status=status,
            size=int(op.total_size),
            n_rhs=1,
            residual_norm=residual_norm,
            relative_residual=residual_norm / rhs_norm,
            max_solution_error=None,
            objective=float(value),
            grad=float(grad),
            finite_difference_grad=float(fd),
            grad_abs_error=abs(float(grad) - float(fd)),
            elapsed_s=float(elapsed_s),
            error=None if status == "ok" else f"Lineax solve result: {result_name}",
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            case="sfincs_tiny_implicit",
            backend="lineax_gmres",
            status="error",
            size=int(op0.total_size),
            n_rhs=1,
            residual_norm=None,
            relative_residual=None,
            max_solution_error=None,
            objective=None,
            grad=None,
            finite_difference_grad=None,
            grad_abs_error=None,
            elapsed_s=None,
            error=str(exc),
        )


def run_current_sfincs_repeated_rhs_gate(
    *,
    input_path: Path,
    tol: float,
    restart: int,
    maxiter: int,
) -> GateResult:
    op0, x_ref, _nu0 = load_tiny_sfincs_fixture(str(input_path))
    restart_use, maxiter_use = _sfincs_gate_solver_window(restart, maxiter)
    perturb = 1.0e-6 * jnp.linspace(1.0, float(op0.total_size), int(op0.total_size), dtype=jnp.float64)
    x_targets = (x_ref, x_ref + perturb)
    rhs_targets = tuple(apply_v3_full_system_operator(op0, x_true) for x_true in x_targets)

    t0 = time.perf_counter()
    x_solutions = [
        linear_custom_solve(
            matvec=lambda v: apply_v3_full_system_operator(op0, v),
            b=rhs,
            tol=tol,
            atol=0.0,
            restart=restart_use,
            maxiter=maxiter_use,
            solver="gmres",
            solve_method="incremental",
            solver_jit=False,
        ).x
        for rhs in rhs_targets
    ]
    for x_sol in x_solutions:
        x_sol.block_until_ready()
    elapsed_s = time.perf_counter() - t0

    residual_norms = [float(jnp.linalg.norm(rhs - apply_v3_full_system_operator(op0, x_sol))) for rhs, x_sol in zip(rhs_targets, x_solutions)]
    rhs_norms = [max(float(jnp.linalg.norm(rhs)), 1.0) for rhs in rhs_targets]
    solution_errors = [float(jnp.linalg.norm(x_sol - x_true)) for x_sol, x_true in zip(x_solutions, x_targets)]
    return GateResult(
        case="sfincs_tiny_repeated_rhs",
        backend="current_custom_linear_solve",
        status="ok",
        size=int(op0.total_size),
        n_rhs=2,
        residual_norm=max(residual_norms),
        relative_residual=max(rn / rhsn for rn, rhsn in zip(residual_norms, rhs_norms)),
        max_solution_error=max(solution_errors),
        objective=None,
        grad=None,
        finite_difference_grad=None,
        grad_abs_error=None,
        elapsed_s=float(elapsed_s),
        error=None,
    )


def run_lineax_sfincs_repeated_rhs_gate(
    *,
    input_path: Path,
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
    op0, x_ref, _nu0 = load_tiny_sfincs_fixture(str(input_path))
    restart_use, maxiter_use = _sfincs_gate_solver_window(restart, maxiter)
    if lx is None:
        return GateResult(
            case="sfincs_tiny_repeated_rhs",
            backend="lineax_gmres",
            status="skipped",
            size=int(op0.total_size),
            n_rhs=2,
            residual_norm=None,
            relative_residual=None,
            max_solution_error=None,
            objective=None,
            grad=None,
            finite_difference_grad=None,
            grad_abs_error=None,
            elapsed_s=None,
            error=f"Lineax unavailable: {import_error}",
        )

    perturb = 1.0e-6 * jnp.linspace(1.0, float(op0.total_size), int(op0.total_size), dtype=jnp.float64)
    x_targets = (x_ref, x_ref + perturb)
    rhs_targets = tuple(apply_v3_full_system_operator(op0, x_true) for x_true in x_targets)
    input_structure = jax.ShapeDtypeStruct(rhs_targets[0].shape, rhs_targets[0].dtype)
    operator = lx.FunctionLinearOperator(lambda v: apply_v3_full_system_operator(op0, v), input_structure)
    solver = lx.GMRES(rtol=tol, atol=0.0, restart=restart_use, max_steps=maxiter_use)

    try:
        state = solver.init(operator, options={})
        t0 = time.perf_counter()
        solutions = [lx.linear_solve(operator, rhs, solver=solver, state=state, throw=False) for rhs in rhs_targets]
        x_solutions = [solution.value for solution in solutions]
        for x_sol in x_solutions:
            x_sol.block_until_ready()
        elapsed_s = time.perf_counter() - t0
        result_names = [_lineax_result_name(lx, solution.result) for solution in solutions]
        residual_norms = [float(jnp.linalg.norm(rhs - apply_v3_full_system_operator(op0, x_sol))) for rhs, x_sol in zip(rhs_targets, x_solutions)]
        rhs_norms = [max(float(jnp.linalg.norm(rhs)), 1.0) for rhs in rhs_targets]
        solution_errors = [float(jnp.linalg.norm(x_sol - x_true)) for x_sol, x_true in zip(x_solutions, x_targets)]
        status = "ok" if all(name == "successful" for name in result_names) else "error"
        return GateResult(
            case="sfincs_tiny_repeated_rhs",
            backend="lineax_gmres",
            status=status,
            size=int(op0.total_size),
            n_rhs=2,
            residual_norm=max(residual_norms),
            relative_residual=max(rn / rhsn for rn, rhsn in zip(residual_norms, rhs_norms)),
            max_solution_error=max(solution_errors),
            objective=None,
            grad=None,
            finite_difference_grad=None,
            grad_abs_error=None,
            elapsed_s=float(elapsed_s),
            error=None if status == "ok" else f"Lineax solve results: {', '.join(result_names)}",
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            case="sfincs_tiny_repeated_rhs",
            backend="lineax_gmres",
            status="error",
            size=int(op0.total_size),
            n_rhs=2,
            residual_norm=None,
            relative_residual=None,
            max_solution_error=None,
            objective=None,
            grad=None,
            finite_difference_grad=None,
            grad_abs_error=None,
            elapsed_s=None,
            error=str(exc),
        )


def run_gate(args: argparse.Namespace) -> list[GateResult]:
    backends = ["current", "lineax"] if args.backend == "all" else [str(args.backend)]
    suites = ["synthetic", "sfincs"] if args.suite == "all" else [str(args.suite)]
    results: list[GateResult] = []
    if "synthetic" in suites:
        matrix, rhs = make_nonsymmetric_system(int(args.size))
        restart = min(int(args.restart), int(args.size))
        if "current" in backends:
            results.append(
                run_current_gate(
                    case="synthetic_nonsymmetric",
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
                    case="synthetic_nonsymmetric",
                    matrix=matrix,
                    rhs=rhs,
                    p0=float(args.shift),
                    tol=float(args.tol),
                    restart=restart,
                    maxiter=int(args.maxiter),
                )
            )
    if "sfincs" in suites:
        input_path = Path(args.input)
        restart = max(1, int(args.restart))
        if "current" in backends:
            results.append(
                run_current_sfincs_implicit_gate(
                    input_path=input_path,
                    tol=float(args.tol),
                    restart=restart,
                    maxiter=int(args.maxiter),
                )
            )
            results.append(
                run_current_sfincs_repeated_rhs_gate(
                    input_path=input_path,
                    tol=float(args.tol),
                    restart=restart,
                    maxiter=int(args.maxiter),
                )
            )
        if "lineax" in backends:
            results.append(
                run_lineax_sfincs_implicit_gate(
                    input_path=input_path,
                    tol=float(args.tol),
                    restart=restart,
                    maxiter=int(args.maxiter),
                )
            )
            results.append(
                run_lineax_sfincs_repeated_rhs_gate(
                    input_path=input_path,
                    tol=float(args.tol),
                    restart=restart,
                    maxiter=int(args.maxiter),
                )
            )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("current", "lineax", "all"), default="all")
    parser.add_argument("--suite", choices=("synthetic", "sfincs", "all"), default="all")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--input", type=Path, default=_default_sfincs_input())
    parser.add_argument("--shift", type=float, default=0.2)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--restart", type=int, default=20)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument(
        "--summary-json",
        type=Path,
        help="Write measured summary and adoption decision JSON without changing --out-json rows.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    results = run_gate(args)
    payload = [result.to_json_dict() for result in results]
    summary = summarize_gate_results(results)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
