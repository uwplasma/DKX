#!/usr/bin/env python
"""Bounded benchmark for RHSMode=1 structured full-system CSR reuse."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_response.full_system import (
    clear_structured_rhs1_full_csr_cache,
    select_structured_rhs1_full_csr_operator,
    solve_structured_rhs1_full_csr,
)
from sfincs_jax.v3_driver import _transport_active_dof_indices
from sfincs_jax.operators.profile_response.system import apply_v3_full_system_operator, full_system_operator_from_namelist, rhs_v3_full_system


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = _REPO_ROOT / "tests" / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
DEFAULT_OUT = _REPO_ROOT / "outputs" / "rhs1_full_csr_reuse_benchmark.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--identity-shift", type=float, default=0.5)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--max-csr-mb", type=float, default=512.0)
    parser.add_argument("--solve", action="store_true", help="Also run explicit host-CSR GMRES on the physical RHS.")
    parser.add_argument("--solve-tol", type=float, default=1.0e-8)
    parser.add_argument("--solve-atol", type=float, default=1.0e-10)
    parser.add_argument("--solve-restart", type=_positive_int, default=80)
    parser.add_argument("--solve-maxiter", type=_positive_int, default=20)
    parser.add_argument(
        "--solve-method",
        choices=("gmres", "lgmres", "direct"),
        default="gmres",
        help="SciPy Krylov method used when --solve is set.",
    )
    parser.add_argument(
        "--active-dof",
        action="store_true",
        help="Solve the active projected RHSMode=1 system and expand back to the full vector.",
    )
    parser.add_argument(
        "--min-residual-reduction",
        type=float,
        default=1.0e-3,
        help="Minimum fractional true-residual reduction for a non-converged solve to pass the promotion gate.",
    )
    parser.add_argument(
        "--preconditioner",
        default="auto",
        help=(
            "Host solve preconditioner: auto, diagonal_schur, xblock_tz_low_l_schur, "
            "block_schur, xi_block_schur, x_xi_block_schur, active_low_l_schur, "
            "active_overlap_schwarz, active_schwarz_low_l_schur, active_xblock, "
            "active_xblock_low_l_schur, active_coarse, active_ilu, jacobi, or none."
        ),
    )
    parser.add_argument("--preconditioner-max-schur-size", type=_positive_int, default=2048)
    parser.add_argument("--preconditioner-max-block-inverse-mb", type=float, default=128.0)
    parser.add_argument(
        "--max-preconditioner-setup-s",
        type=float,
        default=60.0,
        help="Maximum preconditioner setup time allowed by the promotion gate.",
    )
    parser.add_argument(
        "--max-preconditioner-storage-mb",
        type=float,
        default=128.0,
        help="Maximum estimated preconditioner setup storage allowed by the promotion gate.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the benchmark JSON payload.")
    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _benchmark_vector(size: int) -> jax.Array:
    idx = jnp.arange(int(size), dtype=jnp.float64)
    return jnp.sin(0.17 * idx) + 0.25 * jnp.cos(0.31 * idx)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _max_nbytes_from_mb(value: Any) -> int:
    value_f = _finite_float(value)
    if value_f is None:
        return 0
    return int(max(0.0, value_f) * 1024.0 * 1024.0)


def _residual_reduction(initial: Any, final: Any) -> float | None:
    initial_f = _finite_float(initial)
    final_f = _finite_float(final)
    if initial_f is None or final_f is None or initial_f <= 0.0:
        return None
    return float((initial_f - final_f) / initial_f)


def _rate_per_second(value: Any, elapsed_s: Any) -> float | None:
    value_f = _finite_float(value)
    elapsed_f = _finite_float(elapsed_s)
    if value_f is None or elapsed_f is None or elapsed_f <= 0.0:
        return None
    return float(value_f / elapsed_f)


def _history_summary(history: Any) -> dict[str, Any]:
    values = [_finite_float(value) for value in tuple(history or ())]
    finite_values = [float(value) for value in values if value is not None]
    initial = finite_values[0] if finite_values else None
    final = finite_values[-1] if finite_values else None
    reduction = _residual_reduction(initial, final)
    return {
        "values": finite_values,
        "count": int(len(finite_values)),
        "initial": initial,
        "final": final,
        "reduction": reduction,
    }


def _storage_component_bytes(metadata: dict[str, Any], *keys: str) -> dict[str, int]:
    components: dict[str, int] = {}
    for key in keys:
        nbytes = _nonnegative_int(metadata.get(key))
        if nbytes is not None:
            components[key] = int(nbytes)
    return components


def _preconditioner_storage_metadata(
    preconditioner: Any,
    *,
    total_size: Any,
    max_setup_s: Any,
    max_storage_nbytes: int,
) -> dict[str, Any]:
    preconditioner_dict = preconditioner if isinstance(preconditioner, dict) else {}
    metadata_raw = preconditioner_dict.get("metadata", {})
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
    setup_s = _finite_float(preconditioner_dict.get("setup_s"))
    setup_components = _storage_component_bytes(
        metadata,
        "schur_nbytes",
        "block_inverse_nbytes_actual",
        "block_index_nbytes_actual",
        "factor_nbytes_actual",
    )
    diagonal_size = _nonnegative_int(metadata.get("diagonal_size"))
    if diagonal_size is None:
        diagonal_size = _nonnegative_int(metadata.get("kinetic_size"))
    if diagonal_size is not None:
        setup_components["inverse_diagonal_nbytes"] = int(diagonal_size) * np.dtype(np.float64).itemsize
    tail_size = _nonnegative_int(metadata.get("tail_size"))
    if tail_size is not None and "schur_nbytes" in setup_components:
        setup_components["schur_pivot_nbytes_estimate"] = int(tail_size) * np.dtype(np.int32).itemsize

    apply_components = _storage_component_bytes(metadata, "work_vector_nbytes")
    total_size_int = _nonnegative_int(total_size)
    if total_size_int is not None:
        vector_nbytes = int(total_size_int) * np.dtype(np.float64).itemsize
        apply_components["input_vector_nbytes"] = vector_nbytes
        apply_components["output_vector_nbytes"] = vector_nbytes
    kinetic_size = _nonnegative_int(metadata.get("kinetic_size"))
    if kinetic_size is not None:
        apply_components["kinetic_workspace_nbytes_estimate"] = int(kinetic_size) * np.dtype(np.float64).itemsize
    if tail_size is not None:
        apply_components["tail_workspace_nbytes_estimate"] = int(tail_size) * np.dtype(np.float64).itemsize

    setup_storage_nbytes = int(sum(setup_components.values()))
    apply_storage_nbytes = int(sum(apply_components.values()))
    max_setup_s_f = _finite_float(max_setup_s)
    setup_time_threshold_ok = setup_s is None or (
        max_setup_s_f is not None and setup_s <= max_setup_s_f
    )
    setup_storage_threshold_ok = setup_storage_nbytes <= int(max_storage_nbytes)
    return {
        "selected": bool(preconditioner_dict.get("selected", False)),
        "kind": preconditioner_dict.get("kind"),
        "reason": preconditioner_dict.get("reason"),
        "setup": {
            "s": setup_s,
            "max_s": max_setup_s_f,
            "threshold_ok": bool(setup_time_threshold_ok),
            "storage_components_nbytes": setup_components,
            "storage_nbytes_estimate": int(setup_storage_nbytes),
            "max_storage_nbytes": int(max_storage_nbytes),
            "storage_threshold_ok": bool(setup_storage_threshold_ok),
        },
        "apply": {
            "storage_components_nbytes": apply_components,
            "storage_nbytes_estimate": int(apply_storage_nbytes),
        },
        "metadata": metadata,
    }


def _direct_factor_storage_metadata(
    solve_metadata: dict[str, Any],
    *,
    max_setup_s: Any,
    max_storage_nbytes: int,
) -> dict[str, Any]:
    """Return promotion-gate storage metadata for the active direct solve route."""

    factor_s = _finite_float(solve_metadata.get("factor_s"))
    factor_nbytes = _nonnegative_int(solve_metadata.get("factor_nbytes_actual"))
    factor_nnz = _nonnegative_int(solve_metadata.get("factor_nnz"))
    max_setup_s_f = _finite_float(max_setup_s)
    setup_time_threshold_ok = factor_s is not None and max_setup_s_f is not None and factor_s <= max_setup_s_f
    setup_storage_nbytes = int(factor_nbytes or 0)
    setup_storage_threshold_ok = setup_storage_nbytes <= int(max_storage_nbytes)
    components = {}
    if factor_nbytes is not None:
        components["splu_factor_nbytes_actual"] = int(factor_nbytes)
    return {
        "selected": True,
        "kind": solve_metadata.get("factor_kind", "splu"),
        "reason": "direct_solve",
        "setup": {
            "s": factor_s,
            "max_s": max_setup_s_f,
            "threshold_ok": bool(setup_time_threshold_ok),
            "storage_components_nbytes": components,
            "storage_nbytes_estimate": int(setup_storage_nbytes),
            "max_storage_nbytes": int(max_storage_nbytes),
            "storage_threshold_ok": bool(setup_storage_threshold_ok),
        },
        "apply": {
            "storage_components_nbytes": {},
            "storage_nbytes_estimate": 0,
        },
        "metadata": {
            "factor_kind": solve_metadata.get("factor_kind"),
            "factor_nnz": factor_nnz,
            "factor_nbytes_actual": factor_nbytes,
            "permc_spec": solve_metadata.get("permc_spec"),
        },
    }


def _promotion_gate_diagnostics(
    *,
    solve_selected: bool,
    solve_converged: bool,
    residual_reduction: Any,
    min_residual_reduction: Any,
    preconditioner_storage: dict[str, Any],
) -> dict[str, Any]:
    min_reduction = _finite_float(min_residual_reduction)
    reduction = _finite_float(residual_reduction)
    materially_improved = (
        reduction is not None and min_reduction is not None and reduction >= max(0.0, min_reduction)
    )
    setup = preconditioner_storage.get("setup", {}) if isinstance(preconditioner_storage, dict) else {}
    setup_has_measurement = isinstance(setup, dict) and setup.get("s") is not None
    setup_time_ok = bool(setup.get("threshold_ok", False)) if isinstance(setup, dict) else False
    if bool(solve_selected) and not setup_has_measurement:
        setup_time_ok = False
    setup_storage_ok = bool(setup.get("storage_threshold_ok", False)) if isinstance(setup, dict) else False
    reasons: list[str] = []
    if not bool(solve_selected):
        reasons.append("solve_not_selected")
    if not (bool(solve_converged) or materially_improved):
        reasons.append("insufficient_residual_improvement")
    if not setup_time_ok:
        reasons.append("preconditioner_setup_threshold_exceeded")
    if not setup_storage_ok:
        reasons.append("preconditioner_storage_threshold_exceeded")
    passed = not reasons
    return {
        "passed": bool(passed),
        "reasons": reasons,
        "converged": bool(solve_converged),
        "material_residual_improvement": bool(materially_improved),
        "residual_reduction": reduction,
        "min_residual_reduction": min_reduction,
        "setup_threshold_ok": bool(setup_time_ok),
        "storage_threshold_ok": bool(setup_storage_ok),
    }


def _solve_probe_defaults(args: argparse.Namespace) -> dict[str, Any]:
    max_storage_nbytes = _max_nbytes_from_mb(getattr(args, "max_preconditioner_storage_mb", 128.0))
    max_setup_s = _finite_float(getattr(args, "max_preconditioner_setup_s", 60.0))
    min_residual_reduction = _finite_float(getattr(args, "min_residual_reduction", 1.0e-3))
    preconditioner_storage = _preconditioner_storage_metadata(
        {},
        total_size=None,
        max_setup_s=max_setup_s,
        max_storage_nbytes=max_storage_nbytes,
    )
    return {
        "min_residual_reduction": min_residual_reduction,
        "max_preconditioner_setup_s": max_setup_s,
        "max_preconditioner_storage_mb": float(getattr(args, "max_preconditioner_storage_mb", 128.0)),
        "residual_reduction": None,
        "residual_reduction_per_s": None,
        "residual_reduction_per_gmres_s": None,
        "initial_true_residual_norm": None,
        "final_true_residual_norm": None,
        "preconditioned_residual_history": _history_summary(()),
        "initial_preconditioned_residual_norm": None,
        "final_preconditioned_residual_norm": None,
        "preconditioner_storage": preconditioner_storage,
        "promotion_gate": False,
        "promotion_gate_diagnostics": _promotion_gate_diagnostics(
            solve_selected=False,
            solve_converged=False,
            residual_reduction=None,
            min_residual_reduction=min_residual_reduction,
            preconditioner_storage=preconditioner_storage,
        ),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full-system CSR reuse benchmark and return JSON-friendly metrics."""

    input_path = args.input.expanduser().resolve()
    max_preconditioner_storage_nbytes = _max_nbytes_from_mb(
        getattr(args, "max_preconditioner_storage_mb", 128.0)
    )
    payload: dict[str, Any] = {
        "kind": "rhs1_full_csr_reuse_benchmark",
        "input": str(input_path),
        "identity_shift": float(args.identity_shift),
        "repeats": int(args.repeats),
        "max_csr_mb": float(args.max_csr_mb),
        "solve": bool(args.solve),
        "solve_method": str(args.solve_method),
        "active_dof": bool(args.active_dof),
        "preconditioner": str(args.preconditioner),
        "preconditioner_max_schur_size": int(args.preconditioner_max_schur_size),
        "preconditioner_max_block_inverse_mb": float(args.preconditioner_max_block_inverse_mb),
        "dry_run": bool(args.dry_run),
        **_solve_probe_defaults(args),
    }
    if bool(args.dry_run):
        payload["status"] = "planned"
        return payload

    nml = read_sfincs_input(input_path)
    t0 = time.perf_counter()
    op = full_system_operator_from_namelist(nml=nml, identity_shift=float(args.identity_shift))
    operator_build_s = time.perf_counter() - t0

    x = _benchmark_vector(int(op.total_size))
    expected = np.asarray(apply_v3_full_system_operator(op, x)).reshape((-1,))
    jax.block_until_ready(x)

    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    rows: list[dict[str, Any]] = []
    max_csr_nbytes = int(max(0.0, float(args.max_csr_mb)) * 1024.0 * 1024.0)
    for repeat in range(int(args.repeats)):
        select_start = time.perf_counter()
        selected = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=max_csr_nbytes)
        select_s = time.perf_counter() - select_start
        if not bool(selected.selected):
            rows.append(
                {
                    "repeat": int(repeat),
                    "selected": False,
                    "reason": str(selected.reason),
                    "cache_hit": bool(selected.cache_hit),
                    "select_s": float(select_s),
                }
            )
            continue
        matvec_start = time.perf_counter()
        got = selected.matvec(np.asarray(x))
        matvec_s = time.perf_counter() - matvec_start
        rows.append(
            {
                "repeat": int(repeat),
                "selected": True,
                "reason": str(selected.reason),
                "cache_hit": bool(selected.cache_hit),
                "select_s": float(select_s),
                "matvec_s": float(matvec_s),
                "max_abs_error": float(np.max(np.abs(got - expected))),
                "max_rel_error": float(np.max(np.abs(got - expected)) / max(float(np.max(np.abs(expected))), 1.0e-300)),
                "metadata": selected.metadata,
            }
        )

    ok_rows = [row for row in rows if bool(row.get("selected", False))]
    payload.update(
        {
            "status": "ok" if len(ok_rows) == len(rows) else "failed",
            "operator_build_s": float(operator_build_s),
            "total_size": int(op.total_size),
            "rows": rows,
            "max_abs_error": max((float(row["max_abs_error"]) for row in ok_rows), default=None),
            "max_rel_error": max((float(row["max_rel_error"]) for row in ok_rows), default=None),
            "cache_hits": int(sum(1 for row in ok_rows if bool(row.get("cache_hit", False)))),
        }
    )
    if bool(args.solve) and payload["status"] == "ok":
        rhs = rhs_v3_full_system(op)
        initial_true_residual_norm = float(np.linalg.norm(np.asarray(rhs)))
        solve_start = time.perf_counter()
        solve_result = solve_structured_rhs1_full_csr(
            op,
            rhs,
            tol=float(args.solve_tol),
            atol=float(args.solve_atol),
            restart=int(args.solve_restart),
            maxiter=int(args.solve_maxiter),
            method=str(args.solve_method),
            preconditioner=str(args.preconditioner),
            preconditioner_max_schur_size=int(args.preconditioner_max_schur_size),
            preconditioner_max_block_inverse_nbytes=int(
                max(0.0, float(args.preconditioner_max_block_inverse_mb)) * 1024.0 * 1024.0
            ),
            max_csr_nbytes=max_csr_nbytes,
            active_indices=_transport_active_dof_indices(op) if bool(args.active_dof) else None,
        )
        solve_wall_s = time.perf_counter() - solve_start
        true_residual = np.asarray(rhs) - np.asarray(apply_v3_full_system_operator(op, jnp.asarray(solve_result.x)))
        residual_reduction = _residual_reduction(initial_true_residual_norm, solve_result.residual_norm)
        preconditioned_history = _history_summary(solve_result.residual_history)
        solve_result_dict = solve_result.to_dict()
        if str(args.solve_method).strip().lower() == "direct":
            preconditioner_storage = _direct_factor_storage_metadata(
                solve_result.metadata,
                max_setup_s=getattr(args, "max_preconditioner_setup_s", 60.0),
                max_storage_nbytes=max_preconditioner_storage_nbytes,
            )
        else:
            preconditioner_storage = _preconditioner_storage_metadata(
                solve_result.metadata.get("preconditioner", {}),
                total_size=op.total_size,
                max_setup_s=getattr(args, "max_preconditioner_setup_s", 60.0),
                max_storage_nbytes=max_preconditioner_storage_nbytes,
            )
        gate = _promotion_gate_diagnostics(
            solve_selected=bool(solve_result.selection.selected),
            solve_converged=bool(solve_result.converged),
            residual_reduction=residual_reduction,
            min_residual_reduction=getattr(args, "min_residual_reduction", 1.0e-3),
            preconditioner_storage=preconditioner_storage,
        )
        payload["solve_result"] = {
            **solve_result_dict,
            "wrapper_s": float(solve_wall_s),
            "true_residual_norm_jax_operator": float(np.linalg.norm(true_residual)),
        }
        payload.update(
            {
                "initial_true_residual_norm": float(initial_true_residual_norm),
                "final_true_residual_norm": float(solve_result.residual_norm),
                "residual_reduction": residual_reduction,
                "residual_reduction_per_s": _rate_per_second(residual_reduction, solve_wall_s),
                "residual_reduction_per_gmres_s": _rate_per_second(residual_reduction, solve_result.solve_s),
                "preconditioned_residual_history": preconditioned_history,
                "initial_preconditioned_residual_norm": preconditioned_history["initial"],
                "final_preconditioned_residual_norm": preconditioned_history["final"],
                "preconditioner_storage": preconditioner_storage,
                "promotion_gate": bool(gate["passed"]),
                "promotion_gate_diagnostics": gate,
            }
        )
        if not bool(solve_result.converged):
            payload["status"] = "failed"
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_benchmark(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    print(f"Wrote {args.out}")
    return 0 if payload.get("status") in {"ok", "planned"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
