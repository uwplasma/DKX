#!/usr/bin/env python
"""Bounded RHSMode=1 PAS matrix-free correction probe harness.

The script is intentionally opt-in. It probes the existing guarded
``rhs1_pas_matrixfree_correction`` primitive on small deterministic
matrix-free systems and on capped synthetic systems parameterized by
geometry-rich namelist metadata. It does not call the v3 driver or change any
solver defaults.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _REPO_ROOT / "examples" / "performance" / "output" / "rhs1_pas_matrixfree_probe.json"
DEFAULT_METADATA_INPUTS = (
    _REPO_ROOT / "examples" / "sfincs_examples" / "geometryScheme4_2species_PAS_noEr" / "input.namelist",
    _REPO_ROOT / "examples" / "sfincs_examples" / "HSX_PASCollisions_DKESTrajectories" / "input.namelist",
    _REPO_ROOT
    / "examples"
    / "sfincs_examples"
    / "sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_DKESTrajectories"
    / "input.namelist",
)
RESULT_MARKER = "__SFINCS_JAX_RHS1_PAS_MATRIXFREE_RESULT__="
DEFAULT_SYSTEMS = (
    "diagonal_keep",
    "coupled_jacobi_keep",
    "zero_update_reject",
    "nonfinite_candidate_reject",
)
SYSTEMS = DEFAULT_SYSTEMS + ("timeout_sleep",)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe guarded RHSMode=1 PAS matrix-free corrections with hard per-case timeouts."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--systems", nargs="+", default=list(DEFAULT_SYSTEMS), help="Deterministic systems to probe.")
    parser.add_argument(
        "--metadata-inputs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_METADATA_INPUTS),
        help="Optional PAS namelists used only for metadata-parameterized capped synthetic probes.",
    )
    parser.add_argument("--timeout-s", type=_positive_float, default=5.0)
    parser.add_argument("--max-size", type=_positive_int, default=128)
    parser.add_argument("--max-steps", type=_positive_int, default=1)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--min-residual-reduction", type=float, default=1.0e-3)
    parser.add_argument("--block-size", type=_positive_int, default=64)
    parser.add_argument("--max-update-norm-ratio", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true", help="Write the planned probes without running them.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-json", default="", help=argparse.SUPPRESS)
    return parser


def _tail_text(value: str | bytes | None, n: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = str(value)
    return text[-int(n) :]


def _input_record(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def _sanitize_id(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return sanitized or "case"


def _scalar_int(group: dict[str, Any], key: str, default: int) -> int:
    value = group.get(key.upper(), default)
    if isinstance(value, list):
        value = value[0] if value else default
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _scalar_bool(group: dict[str, Any], key: str, default: bool) -> bool:
    value = group.get(key.upper(), default)
    if isinstance(value, list):
        value = value[0] if value else default
    return bool(value)


def _species_count(group: dict[str, Any]) -> int:
    value = group.get("ZS", [])
    if isinstance(value, list):
        return max(1, len(value))
    if value:
        return 1
    return 1


def _read_input_metadata(path: Path) -> dict[str, Any]:
    from sfincs_jax.namelist import read_sfincs_input

    input_path = Path(path)
    nml = read_sfincs_input(input_path)
    resolution = nml.group("resolutionParameters")
    physics = nml.group("physicsParameters")
    geometry = nml.group("geometryParameters")
    species = nml.group("speciesParameters")
    ntheta = _scalar_int(resolution, "NTHETA", 1)
    nzeta = _scalar_int(resolution, "NZETA", 1)
    nxi = _scalar_int(resolution, "NXI", 1)
    nx = _scalar_int(resolution, "NX", 1)
    n_species = _species_count(species)
    geometry_scheme = _scalar_int(geometry, "GEOMETRYSCHEME", -1)
    collision_operator = _scalar_int(physics, "COLLISIONOPERATOR", -1)
    equilibrium_file = geometry.get("EQUILIBRIUMFILE")
    if equilibrium_file is not None:
        equilibrium_file = str(equilibrium_file).strip("\"'")
    return {
        "input": _input_record(input_path),
        "case": input_path.parent.name,
        "geometry_scheme": geometry_scheme,
        "collision_operator": collision_operator,
        "include_phi1": _scalar_bool(physics, "INCLUDEPHI1", False),
        "equilibrium_file": equilibrium_file,
        "resolution": {
            "Ntheta": ntheta,
            "Nzeta": nzeta,
            "Nxi": nxi,
            "Nx": nx,
        },
        "species_count": n_species,
        "estimated_full_unknowns": int(max(1, ntheta) * max(1, nzeta) * max(1, nxi) * max(1, nx) * n_species),
    }


def _config_record(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_steps": int(args.max_steps),
        "omega": float(args.omega),
        "min_residual_reduction": float(args.min_residual_reduction),
        "block_size": int(args.block_size),
        "max_update_norm_ratio": float(args.max_update_norm_ratio),
    }


def _bounded_metadata_size(metadata: dict[str, Any], max_size: int) -> int:
    resolution = metadata["resolution"]
    ntheta = int(resolution["Ntheta"])
    nzeta = int(resolution["Nzeta"])
    nxi = int(resolution["Nxi"])
    nx = int(resolution["Nx"])
    n_species = int(metadata["species_count"])
    raw = ntheta + nzeta + max(1, nxi // 4) + nx * n_species
    return max(4, min(int(max_size), int(raw)))


def _deterministic_case(system: str, args: argparse.Namespace) -> dict[str, Any]:
    size = min(int(args.max_size), 32)
    if system == "diagonal_keep":
        return {
            "case_id": "diagonal_keep",
            "system_kind": "diagonal",
            "size": max(4, min(size, 16)),
            "expected_gate": "keep",
            "expected_reason": "accepted",
            "config": _config_record(args),
        }
    if system == "coupled_jacobi_keep":
        return {
            "case_id": "coupled_jacobi_keep",
            "system_kind": "coupled_jacobi",
            "size": max(4, size),
            "coupling": 0.03,
            "phase": 0.0,
            "expected_gate": "keep",
            "expected_reason": "accepted",
            "config": _config_record(args),
        }
    if system == "zero_update_reject":
        return {
            "case_id": "zero_update_reject",
            "system_kind": "zero_update",
            "size": max(4, min(size, 16)),
            "expected_gate": "reject",
            "expected_reason": "insufficient-residual-improvement",
            "config": _config_record(args),
        }
    if system == "nonfinite_candidate_reject":
        return {
            "case_id": "nonfinite_candidate_reject",
            "system_kind": "nonfinite_candidate",
            "size": max(4, min(size, 16)),
            "expected_gate": "reject",
            "expected_reason": "nonfinite-candidate-residual",
            "config": _config_record(args),
        }
    if system == "timeout_sleep":
        return {
            "case_id": "timeout_sleep",
            "system_kind": "timeout_sleep",
            "size": 1,
            "sleep_s": max(1.0, float(args.timeout_s) * 2.0),
            "expected_gate": "reject",
            "expected_reason": "timeout",
            "config": _config_record(args),
        }
    raise ValueError(f"Unknown deterministic system: {system}")


def build_probe_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    unknown = sorted(set(args.systems) - set(SYSTEMS))
    if unknown:
        raise ValueError(f"Unknown systems: {', '.join(unknown)}")
    for system in args.systems:
        cases.append(_deterministic_case(str(system), args))
    for input_path in args.metadata_inputs:
        metadata = _read_input_metadata(Path(input_path))
        label = _sanitize_id(str(metadata["case"]))
        resolution = metadata["resolution"]
        phase_seed = (
            int(metadata["geometry_scheme"])
            + int(resolution["Ntheta"])
            + 3 * int(resolution["Nzeta"])
            + 5 * int(resolution["Nxi"])
        )
        cases.append(
            {
                "case_id": f"metadata_{label}",
                "system_kind": "metadata_coupled_jacobi",
                "size": _bounded_metadata_size(metadata, int(args.max_size)),
                "coupling": 0.01 + 0.002 * (abs(int(metadata["geometry_scheme"])) % 5),
                "phase": float((phase_seed % 17) / 17.0),
                "expected_gate": "keep",
                "expected_reason": "accepted",
                "config": _config_record(args),
                "source_metadata": metadata,
            }
        )
    return cases


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    cases = build_probe_cases(args)
    return {
        "timeout_s": float(args.timeout_s),
        "dry_run": bool(args.dry_run),
        "max_size": int(args.max_size),
        "systems": list(args.systems),
        "metadata_inputs": [_input_record(Path(path)) for path in args.metadata_inputs],
        "gates": {
            "keep": {
                "requires": [
                    "child status is ok",
                    "correction result is accepted",
                    "residual norms are finite",
                    "observed residual reduction meets min_residual_reduction",
                ]
            },
            "reject": {
                "requires": [
                    "child timeout/error",
                    "non-finite residual/update",
                    "shape mismatch",
                    "insufficient residual improvement",
                ],
                "safe_default": True,
            },
        },
        "cases": cases,
    }


def _json_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _json_float_history(values: tuple[float, ...]) -> list[float | None]:
    return [_json_float(value) for value in values]


def _json_safe_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe_metadata(item) for item in value]
    if isinstance(value, float):
        return _json_float(value)
    return value


def _residual_reduction(initial: float, residual: float) -> float | None:
    initial_f = _json_float(initial)
    residual_f = _json_float(residual)
    if initial_f is None or residual_f is None or initial_f <= 0.0:
        return None
    return float((initial_f - residual_f) / initial_f)


def _gate_from_result(result: Any, config: dict[str, Any], status: str) -> tuple[str, str]:
    if status != "ok":
        return "reject", status
    reduction = _residual_reduction(result.initial_residual_norm, result.residual_norm)
    if (
        bool(result.accepted)
        and result.reason == "accepted"
        and reduction is not None
        and reduction >= float(config["min_residual_reduction"])
    ):
        return "keep", "accepted"
    return "reject", str(result.reason)


def _resource_maxrss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    value = float(usage.ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def _deterministic_vector(jnp: Any, size: int, *, phase: float) -> Any:
    idx = jnp.arange(int(size), dtype=jnp.float32)
    return jnp.sin((idx + 1.0) * (0.13 + 0.01 * phase)) + 0.5 * jnp.cos((idx + 1.0) * (0.07 + 0.02 * phase))


def _build_probe_system(case: dict[str, Any]) -> tuple[Any, Any, Any, Any, dict[str, int]]:
    import jax.numpy as jnp

    counters = {"matvec_calls": 0, "correction_calls": 0}
    size = int(case["size"])
    kind = str(case["system_kind"])
    phase = float(case.get("phase", 0.0))
    x0 = jnp.zeros((size,), dtype=jnp.float32)

    if kind == "timeout_sleep":
        time.sleep(float(case.get("sleep_s", 1.0)))
        rhs = jnp.ones((size,), dtype=jnp.float32)

        def matvec(x: Any) -> Any:
            counters["matvec_calls"] += 1
            return x

        def correction(residual: Any) -> Any:
            counters["correction_calls"] += 1
            return jnp.zeros_like(residual)

        return rhs, x0, matvec, correction, counters

    if kind == "diagonal":
        idx = jnp.arange(size, dtype=jnp.float32)
        diag = 1.0 + 0.5 * ((idx % 7.0) / 6.0)
        x_true = _deterministic_vector(jnp, size, phase=phase)
        rhs = diag * x_true

        def matvec(x: Any) -> Any:
            counters["matvec_calls"] += 1
            return diag * x

        def correction(residual: Any) -> Any:
            counters["correction_calls"] += 1
            return residual / diag

        return rhs, x0, matvec, correction, counters

    if kind in {"coupled_jacobi", "metadata_coupled_jacobi"}:
        idx = jnp.arange(size, dtype=jnp.float32)
        coupling = float(case.get("coupling", 0.03))
        diag = 1.2 + 0.3 * ((idx % 7.0) / 6.0)
        jacobi_diag = diag + jnp.asarray(2.0 * coupling, dtype=jnp.float32)
        x_true = _deterministic_vector(jnp, size, phase=phase)

        def raw_matvec(x: Any) -> Any:
            return diag * x + coupling * (2.0 * x - jnp.roll(x, 1) - jnp.roll(x, -1))

        rhs = raw_matvec(x_true)

        def matvec(x: Any) -> Any:
            counters["matvec_calls"] += 1
            return raw_matvec(x)

        def correction(residual: Any) -> Any:
            counters["correction_calls"] += 1
            return residual / jacobi_diag

        return rhs, x0, matvec, correction, counters

    if kind == "zero_update":
        rhs = _deterministic_vector(jnp, size, phase=phase)

        def matvec(x: Any) -> Any:
            counters["matvec_calls"] += 1
            return x

        def correction(residual: Any) -> Any:
            counters["correction_calls"] += 1
            return jnp.zeros_like(residual)

        return rhs, x0, matvec, correction, counters

    if kind == "nonfinite_candidate":
        rhs = _deterministic_vector(jnp, size, phase=phase)

        def matvec(x: Any) -> Any:
            counters["matvec_calls"] += 1
            if bool(jnp.any(x != 0.0)):
                return jnp.full_like(x, jnp.nan)
            return jnp.zeros_like(x)

        def correction(residual: Any) -> Any:
            counters["correction_calls"] += 1
            return residual

        return rhs, x0, matvec, correction, counters

    raise ValueError(f"Unknown probe system kind: {kind}")


def _child_payload(case: dict[str, Any]) -> dict[str, Any]:
    from sfincs_jax.rhs1_pas_matrixfree import Rhs1PasMatrixFreeConfig, rhs1_pas_matrixfree_correction

    config = dict(case["config"])
    rhs, x0, matvec, correction, counters = _build_probe_system(case)
    t0 = time.perf_counter()
    result = rhs1_pas_matrixfree_correction(
        matvec=matvec,
        rhs=rhs,
        x0=x0,
        correction=correction,
        config=Rhs1PasMatrixFreeConfig(
            max_steps=int(config["max_steps"]),
            omega=float(config["omega"]),
            min_residual_reduction=float(config["min_residual_reduction"]),
            block_size=int(config["block_size"]),
            max_update_norm_ratio=float(config["max_update_norm_ratio"]),
        ),
    )
    elapsed_s = time.perf_counter() - t0
    gate, gate_reason = _gate_from_result(result, config, "ok")
    reduction = _residual_reduction(result.initial_residual_norm, result.residual_norm)
    expected_gate = str(case.get("expected_gate", ""))
    expected_reason = str(case.get("expected_reason", ""))
    return {
        "case_id": str(case["case_id"]),
        "system_kind": str(case["system_kind"]),
        "status": "ok",
        "gate": gate,
        "gate_reason": gate_reason,
        "expected_gate": expected_gate,
        "expected_reason": expected_reason,
        "meets_expected_gate": gate == expected_gate and (not expected_reason or gate_reason == expected_reason),
        "accepted": bool(result.accepted),
        "accepted_steps": int(result.accepted_steps),
        "initial_residual_norm": _json_float(result.initial_residual_norm),
        "residual_norm": _json_float(result.residual_norm),
        "residual_reduction": reduction,
        "residual_history": _json_float_history(result.residual_history),
        "residual_history_nonfinite_count": sum(value is None for value in _json_float_history(result.residual_history)),
        "gate_diagnostics": _json_safe_metadata(result.diagnostics),
        "elapsed_s": float(elapsed_s),
        "max_rss_mb": _resource_maxrss_mb(),
        "metrics": {
            "size": int(case["size"]),
            "matvec_calls": int(counters["matvec_calls"]),
            "correction_calls": int(counters["correction_calls"]),
        },
    }


def _run_child(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    t0 = time.perf_counter()
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--case-json",
        json.dumps(case, sort_keys=True, allow_nan=False),
    ]
    try:
        completed = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=float(args.timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "case_id": str(case["case_id"]),
            "system_kind": str(case["system_kind"]),
            "status": "timeout",
            "gate": "reject",
            "gate_reason": "timeout",
            "expected_gate": str(case.get("expected_gate", "")),
            "expected_reason": str(case.get("expected_reason", "")),
            "meets_expected_gate": str(case.get("expected_gate", "")) == "reject"
            and str(case.get("expected_reason", "")) == "timeout",
            "elapsed_s": float(time.perf_counter() - t0),
            "timeout_s": float(args.timeout_s),
            "stdout_tail": _tail_text(exc.stdout),
            "stderr_tail": _tail_text(exc.stderr),
            "tail_metadata": {
                "stdout_tail_chars": len(_tail_text(exc.stdout)),
                "stderr_tail_chars": len(_tail_text(exc.stderr)),
                "tail_limit_chars": 4000,
            },
        }

    payload: dict[str, Any] | None = None
    for line in completed.stdout.splitlines()[::-1]:
        if line.startswith(RESULT_MARKER):
            payload = json.loads(line[len(RESULT_MARKER) :])
            break
    if payload is None:
        payload = {
            "case_id": str(case["case_id"]),
            "system_kind": str(case["system_kind"]),
            "status": "error",
            "gate": "reject",
            "gate_reason": "missing-result-marker",
            "expected_gate": str(case.get("expected_gate", "")),
            "expected_reason": str(case.get("expected_reason", "")),
            "meets_expected_gate": False,
            "stdout_tail": _tail_text(completed.stdout),
            "stderr_tail": _tail_text(completed.stderr),
        }
    payload["returncode"] = int(completed.returncode)
    payload.setdefault("elapsed_s", float(time.perf_counter() - t0))
    payload.setdefault(
        "tail_metadata",
        {
            "stdout_tail_chars": len(_tail_text(completed.stdout)),
            "stderr_tail_chars": len(_tail_text(completed.stderr)),
            "tail_limit_chars": 4000,
        },
    )
    if completed.returncode != 0:
        payload["status"] = "error"
        payload["gate"] = "reject"
        payload["gate_reason"] = "child-returncode"
        payload["meets_expected_gate"] = False
    return payload


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_gate = {"keep": 0, "reject": 0}
    by_status: dict[str, int] = {}
    unexpected: list[str] = []
    for row in results:
        gate = str(row.get("gate", "reject"))
        by_gate[gate] = by_gate.get(gate, 0) + 1
        status = str(row.get("status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
        if row.get("meets_expected_gate") is False:
            unexpected.append(str(row.get("case_id", "unknown")))
    return {
        "by_gate": by_gate,
        "by_status": by_status,
        "unexpected_cases": unexpected,
        "all_expected_gates_met": not unexpected,
        "lane_state": "harness_only_no_solver_default_change",
        "production_floor_probe_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.child:
        if not args.case_json:
            raise ValueError("--case-json is required in child mode")
        payload = _child_payload(json.loads(args.case_json))
        print(RESULT_MARKER + json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0

    plan = build_plan(args)
    results: list[dict[str, Any]] = []
    if not args.dry_run:
        for case in plan["cases"]:
            row = _run_child(args, case)
            results.append(row)
            print(
                f"{case['case_id']}: {row.get('status')} gate={row.get('gate')} reason={row.get('gate_reason')}",
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "kind": "rhs1_pas_matrixfree_probe",
        "plan": plan,
        "results": results,
        "summary": summarize_results(results),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
