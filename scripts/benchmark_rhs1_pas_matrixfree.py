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
DEFAULT_ARTIFACT_INPUTS = (
    _REPO_ROOT / "examples" / "performance" / "output" / "pas_tz_lgmres_medium_geometry4_probe.json",
    _REPO_ROOT / "examples" / "performance" / "output" / "pas_tz_floor_hsx_dkes_cpu_lgmres_m20_25x51x100x4.json",
    _REPO_ROOT / "examples" / "performance" / "output" / "pas_tz_floor_geom11_cpu_lgmres_m20_25x51x100x4.json",
)
PRODUCTION_FLOOR_TARGETS = ("geometry4", "hsx", "geometry11")
RESULT_MARKER = "__SFINCS_JAX_RHS1_PAS_MATRIXFREE_RESULT__="
DEFAULT_SYSTEMS = (
    "diagonal_keep",
    "coupled_jacobi_keep",
    "zero_update_reject",
    "tiny_update_reject",
    "nonfinite_candidate_reject",
)
SYSTEMS = DEFAULT_SYSTEMS + ("timeout_sleep",)
MAX_DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S = 600.0
DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S = 240.0
DEFAULT_PRODUCTION_SOLVE_VARIANTS = ("tzfft", "tzfft_lgmres")
DEFAULT_PRODUCTION_SOLVE_GRID = {
    "Ntheta": 25,
    "Nzeta": 51,
    "Nxi": 100,
    "Nx": 4,
}
DEFAULT_PRODUCTION_SOLVE_MAX_RSS_MB = 4096.0
DEFAULT_PRODUCTION_SOLVE_MAX_RESIDUAL_NORM = 1.0e-3
DEFAULT_PRODUCTION_SOLVE_MIN_PROMOTION_SPEEDUP = 1.05
DEFAULT_PRODUCTION_SOLVE_MIN_PROMOTION_MEMORY_REDUCTION = 1.05
DEFAULT_BOUNDED_CANDIDATE_ITEMSIZE_BYTES = 4
DEFAULT_PRODUCTION_CANDIDATE_ITEMSIZE_BYTES = 8
DEFAULT_CANDIDATE_LIVE_ARRAYS = 5


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


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and >= 0")
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
    parser.add_argument(
        "--artifact-inputs",
        nargs="*",
        type=Path,
        default=list(DEFAULT_ARTIFACT_INPUTS),
        help="Checked-in PAS benchmark JSON artifacts to inspect for production-floor dry-run evidence.",
    )
    parser.add_argument("--timeout-s", type=_positive_float, default=5.0)
    parser.add_argument("--max-size", type=_positive_int, default=128)
    parser.add_argument("--max-steps", type=_positive_int, default=1)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--min-residual-reduction", type=float, default=1.0e-3)
    parser.add_argument("--block-size", type=_positive_int, default=64)
    parser.add_argument("--max-update-norm-ratio", type=float, default=10.0)
    parser.add_argument("--min-update-norm-ratio", type=_nonnegative_float, default=1.0e-12)
    parser.add_argument("--max-candidate-elements", type=_positive_int, help="Reject candidates above this size.")
    parser.add_argument(
        "--max-candidate-bytes",
        type=_positive_int,
        help="Reject matrix-free candidates whose estimated live vector bytes exceed this limit.",
    )
    parser.add_argument(
        "--run-production-solve-probe",
        action="store_true",
        help="Opt in to bounded production-floor real-solve probes after preflight passes.",
    )
    parser.add_argument(
        "--production-solve-allow-unbudgeted-candidate",
        action="store_true",
        help=(
            "Allow opt-in production real-solve probes without --max-candidate-bytes. "
            "By default real probes fail closed unless the candidate live-vector "
            "memory budget is explicit."
        ),
    )
    parser.add_argument(
        "--production-solve-timeout-s",
        type=_positive_float,
        default=DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S,
        help="Per-target real-solve timeout; defaults below the 10 minute safety cap.",
    )
    parser.add_argument(
        "--production-solve-variants",
        nargs="+",
        default=list(DEFAULT_PRODUCTION_SOLVE_VARIANTS),
        help="PAS-TZ fallback variants for the opt-in real-solve layer.",
    )
    parser.add_argument(
        "--production-solve-targets",
        nargs="+",
        default=list(PRODUCTION_FLOOR_TARGETS),
        help=(
            "Ready production-floor targets to consider for the opt-in real-solve layer. "
            "Accepted case-insensitive names: geometry4, hsx, geometry11."
        ),
    )
    parser.add_argument(
        "--production-solve-total-timeout-s",
        type=_positive_float,
        default=MAX_DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S,
        help="Total parent wall-time budget for all opt-in real-solve targets.",
    )
    parser.add_argument(
        "--production-solve-max-rss-mb",
        type=_nonnegative_float,
        default=DEFAULT_PRODUCTION_SOLVE_MAX_RSS_MB,
        help="Opt-in real-solve RSS gate; 0 records memory without thresholding.",
    )
    parser.add_argument(
        "--production-solve-max-residual-norm",
        type=_nonnegative_float,
        default=DEFAULT_PRODUCTION_SOLVE_MAX_RESIDUAL_NORM,
        help="Opt-in real-solve residual gate.",
    )
    parser.add_argument(
        "--production-solve-expected-backend",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Optional CPU/GPU backend gate passed to the real-solve harness.",
    )
    parser.add_argument(
        "--production-solve-require-default-promotion-gate",
        action="store_true",
        help="Require opt-in real-solve candidates to pass the default-promotion gate.",
    )
    parser.add_argument(
        "--production-solve-baseline-elapsed-s",
        type=_positive_float,
        default=None,
        help="Baseline elapsed time passed to the opt-in real-solve promotion gate.",
    )
    parser.add_argument(
        "--production-solve-baseline-rss-mb",
        type=_positive_float,
        default=None,
        help="Baseline peak RSS passed to the opt-in real-solve promotion gate.",
    )
    parser.add_argument(
        "--production-solve-min-runtime-speedup",
        type=_positive_float,
        default=DEFAULT_PRODUCTION_SOLVE_MIN_PROMOTION_SPEEDUP,
        help="Minimum runtime speedup passed to the opt-in real-solve promotion gate.",
    )
    parser.add_argument(
        "--production-solve-min-memory-reduction",
        type=_positive_float,
        default=DEFAULT_PRODUCTION_SOLVE_MIN_PROMOTION_MEMORY_REDUCTION,
        help="Minimum RSS reduction passed to the opt-in real-solve promotion gate.",
    )
    parser.add_argument(
        "--allow-long-production-solve",
        action="store_true",
        help="Allow opt-in production real-solve timeouts above 600s.",
    )
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
    if isinstance(value, str):
        normalized = value.strip().strip(".").lower()
        if normalized in {"t", "true", "1"}:
            return True
        if normalized in {"f", "false", "0"}:
            return False
    return bool(value)


def _scalar_float(group: dict[str, Any], key: str, default: float) -> float:
    value = group.get(key.upper(), default)
    if isinstance(value, list):
        value = value[0] if value else default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(parsed):
        return float(default)
    return parsed


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
    target = _production_floor_target(
        case=input_path.parent.name,
        geometry_scheme=geometry_scheme,
        equilibrium_file=equilibrium_file,
    )
    return {
        "input": _input_record(input_path),
        "case": input_path.parent.name,
        "source_type": "production_floor_geometry_metadata",
        "production_floor_target": target,
        "geometry_scheme": geometry_scheme,
        "collision_operator": collision_operator,
        "include_phi1": _scalar_bool(physics, "INCLUDEPHI1", False),
        "include_x_dot_term": _scalar_bool(physics, "INCLUDEXDOTTERM", False),
        "include_electric_field_term_in_xi_dot": _scalar_bool(
            physics, "INCLUDEELECTRICFIELDTERMINXIDOT", False
        ),
        "use_dkes_exb_drift": _scalar_bool(physics, "USEDKESEXBDRIFT", False),
        "er": _scalar_float(physics, "ER", 0.0),
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
        "min_update_norm_ratio": float(args.min_update_norm_ratio),
        "max_candidate_elements": None
        if getattr(args, "max_candidate_elements", None) is None
        else int(args.max_candidate_elements),
        "max_candidate_bytes": None
        if getattr(args, "max_candidate_bytes", None) is None
        else int(args.max_candidate_bytes),
    }


def _candidate_byte_preflight_record(
    *,
    element_count: int,
    config: dict[str, Any],
    live_arrays: int = DEFAULT_CANDIDATE_LIVE_ARRAYS,
    itemsize_bytes: int = DEFAULT_BOUNDED_CANDIDATE_ITEMSIZE_BYTES,
    dtype_assumption: str = "float32_bounded_probe",
) -> dict[str, Any]:
    element_count_i = max(0, int(element_count))
    itemsize_i = max(1, int(itemsize_bytes))
    live_arrays_i = max(1, int(live_arrays))
    array_bytes = int(element_count_i * itemsize_i)
    live_bytes = int(array_bytes * live_arrays_i)
    max_elements = config.get("max_candidate_elements")
    max_bytes = config.get("max_candidate_bytes")
    element_budget_configured = max_elements is not None
    byte_budget_configured = max_bytes is not None
    element_safe = (not element_budget_configured) or element_count_i <= int(max_elements)
    byte_safe = (not byte_budget_configured) or live_bytes <= int(max_bytes)
    if not element_safe:
        reason = "candidate-size-limit-exceeded"
    elif not byte_safe:
        reason = "candidate-memory-limit-exceeded"
    elif byte_budget_configured:
        reason = "within-candidate-memory-limit"
    else:
        reason = "candidate-byte-budget-not-configured"
    safe = bool(element_safe and byte_safe)
    return {
        "preflight_kind": "rhs1_pas_matrixfree_candidate",
        "dtype_assumption": dtype_assumption,
        "element_count": element_count_i,
        "itemsize_bytes": itemsize_i,
        "array_bytes": array_bytes,
        "estimated_live_array_count": live_arrays_i,
        "estimated_live_array_bytes": live_bytes,
        "max_candidate_elements": None if max_elements is None else int(max_elements),
        "max_candidate_bytes": None if max_bytes is None else int(max_bytes),
        "candidate_element_budget_configured": bool(element_budget_configured),
        "candidate_byte_budget_configured": bool(byte_budget_configured),
        "candidate_byte_budget_margin": None if max_bytes is None else int(max_bytes) - live_bytes,
        "safe": safe,
        "reason": reason,
        "promotion_gate_passed": bool(safe and byte_budget_configured),
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
    config = _config_record(args)
    if system == "diagonal_keep":
        case_size = max(4, min(size, 16))
        return {
            "case_id": "diagonal_keep",
            "source_type": "synthetic",
            "system_kind": "diagonal",
            "size": case_size,
            "expected_gate": "keep",
            "expected_reason": "accepted",
            "config": config,
            "byte_preflight": _candidate_byte_preflight_record(element_count=case_size, config=config),
        }
    if system == "coupled_jacobi_keep":
        case_size = max(4, size)
        return {
            "case_id": "coupled_jacobi_keep",
            "source_type": "synthetic",
            "system_kind": "coupled_jacobi",
            "size": case_size,
            "coupling": 0.03,
            "phase": 0.0,
            "expected_gate": "keep",
            "expected_reason": "accepted",
            "config": config,
            "byte_preflight": _candidate_byte_preflight_record(element_count=case_size, config=config),
        }
    if system == "zero_update_reject":
        case_size = max(4, min(size, 16))
        return {
            "case_id": "zero_update_reject",
            "source_type": "synthetic",
            "system_kind": "zero_update",
            "size": case_size,
            "expected_gate": "reject",
            "expected_reason": "insufficient-residual-improvement",
            "config": config,
            "byte_preflight": _candidate_byte_preflight_record(element_count=case_size, config=config),
        }
    if system == "tiny_update_reject":
        case_size = max(4, min(size, 16))
        return {
            "case_id": "tiny_update_reject",
            "source_type": "synthetic",
            "system_kind": "tiny_update",
            "size": case_size,
            "update_scale": 1.0e-14,
            "expected_gate": "reject",
            "expected_reason": "update-norm-too-small",
            "config": config,
            "byte_preflight": _candidate_byte_preflight_record(element_count=case_size, config=config),
        }
    if system == "nonfinite_candidate_reject":
        case_size = max(4, min(size, 16))
        return {
            "case_id": "nonfinite_candidate_reject",
            "source_type": "synthetic",
            "system_kind": "nonfinite_candidate",
            "size": case_size,
            "expected_gate": "reject",
            "expected_reason": "nonfinite-candidate-residual",
            "config": config,
            "byte_preflight": _candidate_byte_preflight_record(element_count=case_size, config=config),
        }
    if system == "timeout_sleep":
        return {
            "case_id": "timeout_sleep",
            "source_type": "synthetic",
            "system_kind": "timeout_sleep",
            "size": 1,
            "sleep_s": max(1.0, float(args.timeout_s) * 2.0),
            "expected_gate": "reject",
            "expected_reason": "timeout",
            "config": config,
            "byte_preflight": _candidate_byte_preflight_record(element_count=1, config=config),
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
        config = _config_record(args)
        bounded_size = _bounded_metadata_size(metadata, int(args.max_size))
        phase_seed = (
            int(metadata["geometry_scheme"])
            + int(resolution["Ntheta"])
            + 3 * int(resolution["Nzeta"])
            + 5 * int(resolution["Nxi"])
        )
        cases.append(
            {
                "case_id": f"metadata_{label}",
                "source_type": "production_floor_geometry_metadata",
                "production_floor_target": metadata["production_floor_target"],
                "system_kind": "metadata_coupled_jacobi",
                "size": bounded_size,
                "coupling": 0.01 + 0.002 * (abs(int(metadata["geometry_scheme"])) % 5),
                "phase": float((phase_seed % 17) / 17.0),
                "expected_gate": "keep",
                "expected_reason": "accepted",
                "config": config,
                "byte_preflight": _candidate_byte_preflight_record(element_count=bounded_size, config=config),
                "production_floor_byte_preflight": _candidate_byte_preflight_record(
                    element_count=int(metadata["estimated_full_unknowns"]),
                    config=config,
                    itemsize_bytes=DEFAULT_PRODUCTION_CANDIDATE_ITEMSIZE_BYTES,
                    dtype_assumption="float64_conservative_production_floor",
                ),
                "source_metadata": metadata,
            }
        )
    return cases


def _metadata_input_path(metadata: dict[str, Any]) -> Path:
    raw = Path(str(metadata["input"]))
    return raw if raw.is_absolute() else _REPO_ROOT / raw


def _production_solve_output_path(args: argparse.Namespace, target: str) -> Path:
    return Path(args.out).parent / f"rhs1_pas_production_solve_{target}.json"


def _production_solve_wall_timeout_s(args: argparse.Namespace) -> float:
    timeout_s = float(args.production_solve_timeout_s)
    variant_count = max(1, len(list(args.production_solve_variants)))
    planned = timeout_s * variant_count + 30.0
    if bool(args.allow_long_production_solve):
        return planned
    return min(MAX_DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S, planned)


def _production_solve_total_timeout_s(args: argparse.Namespace) -> float:
    total_timeout_s = float(args.production_solve_total_timeout_s)
    if bool(args.allow_long_production_solve):
        return total_timeout_s
    return min(MAX_DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S, total_timeout_s)


def _normalize_production_solve_target(raw_target: Any) -> str | None:
    target = str(raw_target).strip().lower().replace("-", "_")
    aliases = {
        "geom4": "geometry4",
        "geometry4": "geometry4",
        "geometry_scheme4": "geometry4",
        "geometryscheme4": "geometry4",
        "hsx": "hsx",
        "geom11": "geometry11",
        "geometry11": "geometry11",
        "geometry_scheme11": "geometry11",
        "geometryscheme11": "geometry11",
    }
    return aliases.get(target)


def _requested_production_solve_targets(args: argparse.Namespace) -> list[str]:
    requested: list[str] = []
    seen: set[str] = set()
    invalid: list[str] = []
    for raw_target in getattr(args, "production_solve_targets", PRODUCTION_FLOOR_TARGETS):
        target = _normalize_production_solve_target(raw_target)
        if target is None:
            invalid.append(str(raw_target))
            continue
        if target in seen:
            continue
        requested.append(target)
        seen.add(target)
    if invalid:
        valid = ", ".join(PRODUCTION_FLOOR_TARGETS)
        raise ValueError(f"unknown production solve target(s): {', '.join(invalid)}; expected one of {valid}")
    return requested or list(PRODUCTION_FLOOR_TARGETS)


def _validate_production_solve_bounds(args: argparse.Namespace) -> None:
    _requested_production_solve_targets(args)
    if (
        bool(getattr(args, "run_production_solve_probe", False))
        and not bool(getattr(args, "production_solve_allow_unbudgeted_candidate", False))
        and getattr(args, "max_candidate_bytes", None) is None
    ):
        raise ValueError(
            "--max-candidate-bytes is required with --run-production-solve-probe "
            "unless --production-solve-allow-unbudgeted-candidate is set"
        )
    if bool(getattr(args, "production_solve_require_default_promotion_gate", False)):
        if getattr(args, "production_solve_baseline_elapsed_s", None) is None:
            raise ValueError(
                "--production-solve-baseline-elapsed-s is required with "
                "--production-solve-require-default-promotion-gate"
            )
        if getattr(args, "production_solve_baseline_rss_mb", None) is None:
            raise ValueError(
                "--production-solve-baseline-rss-mb is required with "
                "--production-solve-require-default-promotion-gate"
            )
        if getattr(args, "max_candidate_bytes", None) is None:
            raise ValueError(
                "--max-candidate-bytes is required with "
                "--production-solve-require-default-promotion-gate"
            )
    if bool(args.allow_long_production_solve):
        return
    timeout_s = float(args.production_solve_timeout_s)
    if timeout_s > MAX_DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S:
        raise ValueError(
            "default production real-solve probes are capped at 600s; "
            "pass --allow-long-production-solve for explicit longer probes"
        )
    total_timeout_s = float(args.production_solve_total_timeout_s)
    if total_timeout_s > MAX_DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S:
        raise ValueError(
            "default production real-solve probe batches are capped at 600s total; "
            "pass --allow-long-production-solve for explicit longer batches"
        )


def build_bounded_real_solve_probe(
    args: argparse.Namespace,
    *,
    cases: list[dict[str, Any]],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Build the opt-in production-floor real-solve probe plan."""
    target_cases = {
        target: _metadata_matches(cases, target)
        for target in PRODUCTION_FLOOR_TARGETS
    }
    targets: dict[str, Any] = {}
    requested_targets = _requested_production_solve_targets(args)
    per_target_timeout_s = _production_solve_wall_timeout_s(args)
    total_timeout_s = _production_solve_total_timeout_s(args)
    planned_wall_timeout_s = 0.0
    selected_targets: list[str] = []
    for target in PRODUCTION_FLOOR_TARGETS:
        preflight_target = dict(preflight.get("targets", {}).get(target, {}))
        preflight_gates = preflight_target.get("gates", {})
        if not isinstance(preflight_gates, dict):
            preflight_gates = {}
        byte_preflight_gate = preflight_gates.get("candidate_byte_preflight", {})
        if not isinstance(byte_preflight_gate, dict):
            byte_preflight_gate = {}
        default_promotion_required = bool(args.production_solve_require_default_promotion_gate)
        byte_preflight_required = bool(
            default_promotion_required
            or (
                bool(args.run_production_solve_probe)
                and not bool(args.production_solve_allow_unbudgeted_candidate)
            )
        )
        byte_preflight_promotion_ready = (
            not byte_preflight_required
            or (
                byte_preflight_gate.get("status") == "pass"
                and byte_preflight_gate.get("promotion_gate_passed") is True
            )
        )
        metadata_cases = target_cases[target]
        metadata = metadata_cases[0].get("source_metadata") if metadata_cases else None
        ready = bool(preflight_target.get("ready") is True and metadata)
        out_path = _production_solve_output_path(args, target)
        command: list[str] = []
        if metadata:
            input_path = _metadata_input_path(metadata)
            command = [
                sys.executable,
                str(_REPO_ROOT / "scripts" / "benchmark_pas_tz_memory_fallback.py"),
                "--input",
                str(input_path),
                "--out",
                str(out_path),
                "--variants",
                *(str(variant) for variant in args.production_solve_variants),
                "--timeout-s",
                str(float(args.production_solve_timeout_s)),
                "--stall-s",
                str(float(args.production_solve_timeout_s)),
                "--maxiter",
                "20",
                "--restart",
                "20",
                "--solve-method",
                "incremental",
                "--Ntheta",
                str(DEFAULT_PRODUCTION_SOLVE_GRID["Ntheta"]),
                "--Nzeta",
                str(DEFAULT_PRODUCTION_SOLVE_GRID["Nzeta"]),
                "--Nxi",
                str(DEFAULT_PRODUCTION_SOLVE_GRID["Nxi"]),
                "--Nx",
                str(DEFAULT_PRODUCTION_SOLVE_GRID["Nx"]),
                "--max-rss-mb",
                str(float(args.production_solve_max_rss_mb)),
                "--max-residual-norm",
                str(float(args.production_solve_max_residual_norm)),
                "--expected-backend",
                str(args.production_solve_expected_backend),
            ]
            if bool(args.production_solve_require_default_promotion_gate):
                command.extend(
                    [
                        "--require-default-promotion-gate",
                        "--baseline-elapsed-s",
                        str(float(args.production_solve_baseline_elapsed_s)),
                        "--baseline-rss-mb",
                        str(float(args.production_solve_baseline_rss_mb)),
                        "--min-runtime-speedup",
                        str(float(args.production_solve_min_runtime_speedup)),
                        "--min-memory-reduction",
                        str(float(args.production_solve_min_memory_reduction)),
                    ]
                )
            if bool(args.allow_long_production_solve):
                command.append("--allow-long-run")
        requested = target in requested_targets
        budget_fit = planned_wall_timeout_s + per_target_timeout_s <= total_timeout_s
        selected_by_budget = bool(ready and requested and budget_fit and byte_preflight_promotion_ready)
        if selected_by_budget:
            planned_wall_timeout_s += per_target_timeout_s
            selected_targets.append(target)
        if not ready:
            skip_reason = preflight_target.get("next_solve_recommendation", "preflight-not-ready")
        elif not requested:
            skip_reason = "not-requested"
        elif not byte_preflight_promotion_ready:
            skip_reason = "candidate-byte-preflight-not-promotion-safe"
        elif not budget_fit:
            skip_reason = "production-solve-total-timeout-budget-exhausted"
        else:
            skip_reason = None
        targets[target] = {
            "ready": ready,
            "requested": requested,
            "budget_fit": budget_fit,
            "byte_preflight_required": byte_preflight_required,
            "byte_preflight_promotion_ready": byte_preflight_promotion_ready,
            "candidate_byte_preflight": byte_preflight_gate,
            "planned_parent_timeout_s": per_target_timeout_s,
            "total_wall_timeout_budget_s": total_timeout_s,
            "selected_by_budget": selected_by_budget,
            "will_run": bool(args.run_production_solve_probe and (not args.dry_run) and selected_by_budget),
            "skip_reason": skip_reason,
            "input": str(_metadata_input_path(metadata)) if metadata else None,
            "out": str(out_path),
            "command": command,
        }
    return {
        "mode": "bounded_real_solve_probe",
        "description": "Opt-in subprocess layer using benchmark_pas_tz_memory_fallback.py; default planning launches no solves.",
        "run_requested": bool(args.run_production_solve_probe),
        "dry_run": bool(args.dry_run),
        "max_default_runtime_s": MAX_DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S,
        "timeout_s": float(args.production_solve_timeout_s),
        "parent_wall_timeout_s": per_target_timeout_s,
        "total_wall_timeout_budget_s": total_timeout_s,
        "planned_wall_timeout_s": planned_wall_timeout_s,
        "selected_targets": selected_targets,
        "requested_targets": requested_targets,
        "selection_policy": "ready requested targets in production-floor order until the total wall-time budget is exhausted",
        "safety_policy": {
            "requires_explicit_run_flag": True,
            "dry_run_launches_subprocesses": False,
            "default_batch_cap_s": MAX_DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S,
            "invalid_targets_fail_closed": True,
            "requires_candidate_byte_budget_by_default": True,
            "allow_unbudgeted_candidate": bool(args.production_solve_allow_unbudgeted_candidate),
        },
        "allow_long_run": bool(args.allow_long_production_solve),
        "variants": list(args.production_solve_variants),
        "grid_overrides": dict(DEFAULT_PRODUCTION_SOLVE_GRID),
        "gates": {
            "stall_s": float(args.production_solve_timeout_s),
            "max_rss_mb": float(args.production_solve_max_rss_mb),
            "max_residual_norm": float(args.production_solve_max_residual_norm),
            "expected_backend": str(args.production_solve_expected_backend),
            "solver_path_churn_allowed": False,
            "default_promotion_required": bool(args.production_solve_require_default_promotion_gate),
            "candidate_byte_preflight_required": bool(
                args.production_solve_require_default_promotion_gate
                or (
                    bool(args.run_production_solve_probe)
                    and not bool(args.production_solve_allow_unbudgeted_candidate)
                )
            ),
            "baseline_elapsed_s": _json_float(args.production_solve_baseline_elapsed_s),
            "baseline_rss_mb": _json_float(args.production_solve_baseline_rss_mb),
            "min_runtime_speedup": float(args.production_solve_min_runtime_speedup),
            "min_memory_reduction": float(args.production_solve_min_memory_reduction),
            "promotion_policy": (
                "real-solve candidates must not regress elapsed_s or max_rss_mb "
                "against baseline when default promotion is required"
            ),
        },
        "targets": targets,
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    cases = build_probe_cases(args)
    artifact_probe = build_artifact_probe(args.artifact_inputs)
    preflight = build_production_floor_preflight(cases=cases, artifacts=artifact_probe["artifacts"])
    real_solve_probe = build_bounded_real_solve_probe(args, cases=cases, preflight=preflight)
    return {
        "timeout_s": float(args.timeout_s),
        "dry_run": bool(args.dry_run),
        "max_size": int(args.max_size),
        "systems": list(args.systems),
        "metadata_inputs": [_input_record(Path(path)) for path in args.metadata_inputs],
        "artifact_inputs": [_input_record(Path(path)) for path in args.artifact_inputs],
        "artifact_probe": artifact_probe,
        "production_floor_preflight": preflight,
        "bounded_real_solve_probe": real_solve_probe,
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
                    "candidate element or byte preflight budget exceeded",
                    "insufficient residual improvement",
                ],
                "safe_default": True,
            },
        },
        "cases": cases,
    }


def _production_floor_target(
    *,
    case: str | None = None,
    geometry_scheme: int | None = None,
    equilibrium_file: str | None = None,
    text: str | None = None,
) -> str:
    haystack = " ".join(str(item or "") for item in (case, equilibrium_file, text)).lower()
    if "hsx" in haystack:
        return "hsx"
    if geometry_scheme == 4 or "geometryscheme4" in haystack or "geometry4" in haystack:
        return "geometry4"
    if geometry_scheme == 11 or "geometryscheme11" in haystack or "geom11" in haystack:
        return "geometry11"
    return "unknown"


def _extract_message_float(pattern: str, messages: list[str]) -> float | None:
    compiled = re.compile(pattern)
    for message in messages:
        match = compiled.search(str(message))
        if match:
            return _json_float(match.group(1))
    return None


def _read_artifact_probe(path: Path) -> dict[str, Any]:
    input_path = Path(path)
    record: dict[str, Any] = {
        "path": _input_record(input_path),
        "status": "missing",
        "target": _production_floor_target(text=str(input_path)),
        "ready_evidence": False,
    }
    if not input_path.exists():
        return record
    try:
        payload = json.loads(input_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        record.update({"status": "error", "error": type(exc).__name__})
        return record
    if not isinstance(payload, dict):
        record.update({"status": "error", "error": "artifact-root-not-object"})
        return record

    plan = payload.get("plan", {})
    results = payload.get("results", [])
    if not isinstance(plan, dict):
        plan = {}
    if not isinstance(results, list):
        results = []
    plan_input = str(plan.get("input", ""))
    target = _production_floor_target(text=f"{input_path} {plan_input}")
    ok_results = [row for row in results if isinstance(row, dict) and row.get("status") == "ok"]
    residual_norms = [_json_float(row.get("residual_norm")) for row in ok_results]
    finite_residual_norms = [value for value in residual_norms if value is not None]
    messages = [
        str(message)
        for row in ok_results
        for message in row.get("messages_tail", [])
        if isinstance(row.get("messages_tail", []), list)
    ]
    guarded_pas_tz_seen = any("PAS-TZ guarded" in message for message in messages)
    total_size_max = max(
        (
            value
            for value in (
                _extract_message_float(r"total_size=([0-9.eE+-]+)", [message]) for message in messages
            )
            if value is not None
        ),
        default=None,
    )
    pas_constraint_size_max = max(
        (
            value
            for value in (
                _extract_message_float(r"PAS constraint projection enabled \(size=([0-9.eE+-]+)", [message])
                for message in messages
            )
            if value is not None
        ),
        default=None,
    )
    max_rss_values = [_json_float(row.get("max_rss_mb")) for row in ok_results]
    elapsed_values = [_json_float(row.get("elapsed_s")) for row in ok_results]
    best_residual_norm = min(finite_residual_norms) if finite_residual_norms else None
    max_rss_mb_peak = max((value for value in max_rss_values if value is not None), default=None)
    elapsed_s_max = max((value for value in elapsed_values if value is not None), default=None)
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    failed_variants = summary.get("failed_variants")
    if not isinstance(failed_variants, list):
        failed_variants = []
    promotion_eligible_variants = summary.get("promotion_eligible_variants")
    if not isinstance(promotion_eligible_variants, list):
        promotion_eligible_variants = []
    failure_reasons = summary.get("failure_reasons")
    if not isinstance(failure_reasons, dict):
        failure_reasons = {}
    artifact_gates = {
        "guarded_pas_tz": _gate(
            "pass" if guarded_pas_tz_seen else "fail",
            "guarded PAS-TZ evidence recorded" if guarded_pas_tz_seen else "missing guarded PAS-TZ evidence",
        ),
        "residual": _gate(
            "pass"
            if best_residual_norm is not None
            and best_residual_norm <= DEFAULT_PRODUCTION_SOLVE_MAX_RESIDUAL_NORM
            else "fail",
            "residual-clean"
            if best_residual_norm is not None
            and best_residual_norm <= DEFAULT_PRODUCTION_SOLVE_MAX_RESIDUAL_NORM
            else "missing-or-high-residual",
            best_residual_norm=best_residual_norm,
            max_residual_norm=DEFAULT_PRODUCTION_SOLVE_MAX_RESIDUAL_NORM,
        ),
        "runtime": _gate(
            "pass"
            if elapsed_s_max is not None
            and elapsed_s_max <= DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S
            else "fail",
            "runtime-within-short-probe-bound"
            if elapsed_s_max is not None
            and elapsed_s_max <= DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S
            else "missing-or-slow-runtime",
            elapsed_s_max=elapsed_s_max,
            limit_s=DEFAULT_PRODUCTION_SOLVE_TIMEOUT_S,
        ),
        "memory": _gate(
            "pass"
            if max_rss_mb_peak is not None
            and max_rss_mb_peak <= DEFAULT_PRODUCTION_SOLVE_MAX_RSS_MB
            else "fail",
            "rss-within-short-probe-bound"
            if max_rss_mb_peak is not None
            and max_rss_mb_peak <= DEFAULT_PRODUCTION_SOLVE_MAX_RSS_MB
            else "missing-or-high-rss",
            max_rss_mb_peak=max_rss_mb_peak,
            limit_mb=DEFAULT_PRODUCTION_SOLVE_MAX_RSS_MB,
        ),
        "row_gates": _gate(
            "pass" if summary.get("all_gates_passed") is True else "fail",
            "artifact row gates passed"
            if summary.get("all_gates_passed") is True
            else "missing-artifact-row-gate-summary"
            if "all_gates_passed" not in summary
            else "artifact row gates failed",
            all_gates_passed=summary.get("all_gates_passed"),
            failed_variants=failed_variants,
            promotion_eligible_variants=promotion_eligible_variants,
            failure_reasons=failure_reasons,
        ),
    }
    artifact_gates_passed = all(gate["status"] == "pass" for gate in artifact_gates.values())
    record.update(
        {
            "status": "ok",
            "kind": str(payload.get("kind", "")),
            "target": target,
            "plan_input": plan_input,
            "variants": list(plan.get("variants", [])) if isinstance(plan.get("variants", []), list) else [],
            "result_count": len(results),
            "ok_result_count": len(ok_results),
            "finite_residual_count": len(finite_residual_norms),
            "best_residual_norm": best_residual_norm,
            "max_rss_mb_peak": max_rss_mb_peak,
            "elapsed_s_max": elapsed_s_max,
            "guarded_pas_tz_seen": guarded_pas_tz_seen,
            "total_size_max": total_size_max,
            "pas_constraint_size_max": pas_constraint_size_max,
            "artifact_gates": artifact_gates,
            "artifact_gates_passed": artifact_gates_passed,
            "ready_evidence": bool(ok_results and artifact_gates_passed),
        }
    )
    return _json_safe_metadata(record)


def build_artifact_probe(paths: list[Path]) -> dict[str, Any]:
    artifacts = [_read_artifact_probe(Path(path)) for path in paths]
    by_target = {target: 0 for target in PRODUCTION_FLOOR_TARGETS}
    for artifact in artifacts:
        target = str(artifact.get("target", "unknown"))
        if target in by_target and artifact.get("ready_evidence") is True:
            by_target[target] += 1
    return {
        "mode": "checked_in_artifact_dry_run",
        "description": "Inspects existing PAS benchmark JSON; does not launch production solves.",
        "artifacts": artifacts,
        "ready_evidence_by_target": by_target,
    }


def _gate(status: str, reason: str, **extra: Any) -> dict[str, Any]:
    payload = {"status": status, "reason": reason}
    payload.update(extra)
    return _json_safe_metadata(payload)


def _metadata_matches(cases: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    return [
        case
        for case in cases
        if case.get("source_type") == "production_floor_geometry_metadata"
        and case.get("production_floor_target") == target
    ]


def _artifact_matches(artifacts: list[dict[str, Any]], target: str) -> list[dict[str, Any]]:
    return [artifact for artifact in artifacts if artifact.get("target") == target]


def build_production_floor_preflight(
    *, cases: list[dict[str, Any]], artifacts: list[dict[str, Any]]
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    for target in PRODUCTION_FLOOR_TARGETS:
        target_cases = _metadata_matches(cases, target)
        target_artifacts = _artifact_matches(artifacts, target)
        metadata = [case["source_metadata"] for case in target_cases if "source_metadata" in case]
        artifact_ready = [artifact for artifact in target_artifacts if artifact.get("ready_evidence") is True]
        byte_records = [
            case.get("production_floor_byte_preflight", case.get("byte_preflight", {}))
            for case in target_cases
            if isinstance(case.get("production_floor_byte_preflight", case.get("byte_preflight", {})), dict)
        ]
        unsafe_byte_records = [record for record in byte_records if record.get("safe") is False]
        byte_budget_configured = any(record.get("candidate_byte_budget_configured") is True for record in byte_records)
        byte_promotion_ready = bool(byte_records and byte_budget_configured and not unsafe_byte_records)
        gates = {
            "metadata_present": _gate(
                "pass" if metadata else "fail",
                "production-floor namelist metadata found" if metadata else "no matching production-floor metadata",
                count=len(metadata),
            ),
            "pas_collision_operator": _gate(
                "pass" if metadata and all(item.get("collision_operator") == 1 for item in metadata) else "fail",
                "all matching metadata uses PAS collisions"
                if metadata and all(item.get("collision_operator") == 1 for item in metadata)
                else "missing metadata or non-PAS collision operator",
                observed=[item.get("collision_operator") for item in metadata],
            ),
            "bounded_matrixfree_case": _gate(
                "pass" if target_cases and all(int(case.get("size", 0)) >= 4 for case in target_cases) else "fail",
                "bounded metadata-parameterized matrix-free probe case planned"
                if target_cases
                else "no bounded metadata-parameterized probe case planned",
                planned_case_ids=[case.get("case_id") for case in target_cases],
                planned_sizes=[case.get("size") for case in target_cases],
            ),
            "candidate_byte_preflight": _gate(
                "pass" if byte_records and not unsafe_byte_records else "fail",
                "candidate byte preflight is within configured budget"
                if byte_promotion_ready
                else "candidate byte preflight recorded without byte budget; diagnostic only"
                if byte_records and not byte_budget_configured
                else "candidate byte preflight exceeds configured budget"
                if unsafe_byte_records
                else "missing candidate byte preflight",
                byte_budget_configured=byte_budget_configured,
                promotion_gate_passed=byte_promotion_ready,
                unsafe_count=len(unsafe_byte_records),
                records=byte_records,
            ),
            "checked_artifact_evidence": _gate(
                "pass" if artifact_ready else "fail",
                "checked-in PAS benchmark artifact has guarded residual/runtime/memory evidence"
                if artifact_ready
                else "no checked-in artifact with guarded residual/runtime/memory evidence",
                artifact_paths=[artifact.get("path") for artifact in target_artifacts],
                ready_artifact_paths=[artifact.get("path") for artifact in artifact_ready],
            ),
            "long_solve_avoidance": _gate(
                "pass",
                "preflight inspects metadata and checked-in artifacts only; production solves are not launched",
            ),
        }
        ready = all(gate["status"] == "pass" for gate in gates.values())
        next_solve_recommendation = (
            "proceed_to_short_real_solve_probe"
            if ready
            else "candidate-byte-preflight-not-promotion-safe"
            if unsafe_byte_records
            else "hold_for_missing_evidence"
        )
        targets[target] = {
            "ready": ready,
            "gates": gates,
            "next_solve_recommendation": next_solve_recommendation,
        }
    ready_targets = [target for target, record in targets.items() if record["ready"]]
    return {
        "mode": "production_floor_preflight",
        "required_targets": list(PRODUCTION_FLOOR_TARGETS),
        "ready_targets": ready_targets,
        "all_required_targets_ready": len(ready_targets) == len(PRODUCTION_FLOOR_TARGETS),
        "targets": targets,
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

    if kind == "tiny_update":
        rhs = _deterministic_vector(jnp, size, phase=phase)
        update_scale = float(case.get("update_scale", 1.0e-14))

        def matvec(x: Any) -> Any:
            counters["matvec_calls"] += 1
            return x

        def correction(residual: Any) -> Any:
            counters["correction_calls"] += 1
            return jnp.asarray(update_scale, dtype=residual.dtype) * residual

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
            min_update_norm_ratio=float(config["min_update_norm_ratio"]),
            max_candidate_elements=None
            if config.get("max_candidate_elements") is None
            else int(config["max_candidate_elements"]),
            max_candidate_bytes=None
            if config.get("max_candidate_bytes") is None
            else int(config["max_candidate_bytes"]),
        ),
    )
    elapsed_s = time.perf_counter() - t0
    gate, gate_reason = _gate_from_result(result, config, "ok")
    reduction = _residual_reduction(result.initial_residual_norm, result.residual_norm)
    expected_gate = str(case.get("expected_gate", ""))
    expected_reason = str(case.get("expected_reason", ""))
    return {
        "case_id": str(case["case_id"]),
        "source_type": str(case.get("source_type", "synthetic")),
        "production_floor_target": str(case.get("production_floor_target", "")),
        "system_kind": str(case["system_kind"]),
        "byte_preflight": _json_safe_metadata(case.get("byte_preflight", {})),
        "production_floor_byte_preflight": _json_safe_metadata(
            case.get("production_floor_byte_preflight", {})
        ),
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
            "source_type": str(case.get("source_type", "synthetic")),
            "production_floor_target": str(case.get("production_floor_target", "")),
            "system_kind": str(case["system_kind"]),
            "byte_preflight": _json_safe_metadata(case.get("byte_preflight", {})),
            "production_floor_byte_preflight": _json_safe_metadata(
                case.get("production_floor_byte_preflight", {})
            ),
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
            "source_type": str(case.get("source_type", "synthetic")),
            "production_floor_target": str(case.get("production_floor_target", "")),
            "system_kind": str(case["system_kind"]),
            "byte_preflight": _json_safe_metadata(case.get("byte_preflight", {})),
            "production_floor_byte_preflight": _json_safe_metadata(
                case.get("production_floor_byte_preflight", {})
            ),
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


def _load_probe_summary(out_path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(out_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    summary = payload.get("summary")
    return summary if isinstance(summary, dict) else None


def run_bounded_real_solve_probe(
    args: argparse.Namespace,
    probe_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run ready opt-in production-floor real-solve probes as bounded subprocesses."""
    results: list[dict[str, Any]] = []
    wall_timeout_s = float(probe_plan["parent_wall_timeout_s"])
    batch_timeout_s = float(probe_plan.get("total_wall_timeout_budget_s", wall_timeout_s))
    batch_deadline = time.perf_counter() + batch_timeout_s
    targets = probe_plan.get("targets", {})
    if not isinstance(targets, dict):
        return results
    for target, record in targets.items():
        if not isinstance(record, dict) or not record.get("will_run"):
            continue
        command = record.get("command", [])
        if not isinstance(command, list) or not command:
            results.append({"target": str(target), "status": "error", "reason": "missing-command"})
            continue
        remaining_batch_s = batch_deadline - time.perf_counter()
        if remaining_batch_s <= 0.0:
            results.append(
                {
                    "target": str(target),
                    "status": "skipped",
                    "reason": "production-solve-total-timeout-budget-exhausted",
                    "timeout_s": 0.0,
                    "out": record.get("out"),
                }
            )
            continue
        target_timeout_s = min(wall_timeout_s, remaining_batch_s)
        t0 = time.perf_counter()
        try:
            completed = subprocess.run(
                [str(item) for item in command],
                text=True,
                capture_output=True,
                timeout=target_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            results.append(
                {
                    "target": str(target),
                    "status": "timeout",
                    "elapsed_s": float(time.perf_counter() - t0),
                    "timeout_s": target_timeout_s,
                    "batch_timeout_s": batch_timeout_s,
                    "out": record.get("out"),
                    "stdout_tail": _tail_text(exc.stdout),
                    "stderr_tail": _tail_text(exc.stderr),
                }
            )
            continue
        out_path = Path(str(record.get("out", "")))
        summary = _load_probe_summary(out_path)
        results.append(
            {
                "target": str(target),
                "status": "ok" if completed.returncode == 0 else "error",
                "returncode": int(completed.returncode),
                "elapsed_s": float(time.perf_counter() - t0),
                "timeout_s": target_timeout_s,
                "batch_timeout_s": batch_timeout_s,
                "out": str(out_path),
                "summary": summary,
                "all_gates_passed": bool(summary and summary.get("all_gates_passed") is True),
                "stdout_tail": _tail_text(completed.stdout),
                "stderr_tail": _tail_text(completed.stderr),
            }
        )
    return results


def summarize_results(
    results: list[dict[str, Any]],
    preflight: dict[str, Any] | None = None,
    production_solve_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
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
    preflight_ready = bool(preflight and preflight.get("all_required_targets_ready") is True)
    solve_results = production_solve_results or []
    solve_failures = [
        str(row.get("target", "unknown"))
        for row in solve_results
        if row.get("status") != "ok" or row.get("all_gates_passed") is False
    ]
    byte_preflight_targets: dict[str, Any] = {}
    if preflight and isinstance(preflight.get("targets"), dict):
        for target, record in preflight["targets"].items():
            if not isinstance(record, dict):
                continue
            gates = record.get("gates", {})
            if not isinstance(gates, dict):
                continue
            byte_gate = gates.get("candidate_byte_preflight", {})
            if not isinstance(byte_gate, dict):
                continue
            byte_preflight_targets[str(target)] = {
                "status": byte_gate.get("status"),
                "byte_budget_configured": byte_gate.get("byte_budget_configured"),
                "promotion_gate_passed": byte_gate.get("promotion_gate_passed"),
                "unsafe_count": byte_gate.get("unsafe_count"),
            }
    return {
        "by_gate": by_gate,
        "by_status": by_status,
        "unexpected_cases": unexpected,
        "all_expected_gates_met": not unexpected,
        "lane_state": "harness_only_no_solver_default_change",
        "candidate_byte_preflight": {
            "targets": byte_preflight_targets,
            "promotion_safe_target_count": sum(
                1 for record in byte_preflight_targets.values() if record.get("promotion_gate_passed") is True
            ),
            "unsafe_target_count": sum(
                1 for record in byte_preflight_targets.values() if int(record.get("unsafe_count") or 0) > 0
            ),
        },
        "production_floor_probe_ready": preflight_ready and (not results or not unexpected),
        "production_real_solve_result_count": len(solve_results),
        "production_real_solve_failures": solve_failures,
        "production_real_solve_all_gates_passed": bool(solve_results) and not solve_failures,
        "next_real_solve_recommendation": "proceed_to_short_real_solve_probe"
        if preflight_ready and (not results or not unexpected)
        else "hold_for_missing_or_unexpected_probe_evidence",
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_production_solve_bounds(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.child:
        if not args.case_json:
            raise ValueError("--case-json is required in child mode")
        payload = _child_payload(json.loads(args.case_json))
        print(RESULT_MARKER + json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0

    plan = build_plan(args)
    results: list[dict[str, Any]] = []
    production_solve_results: list[dict[str, Any]] = []
    if not args.dry_run:
        for case in plan["cases"]:
            row = _run_child(args, case)
            results.append(row)
            print(
                f"{case['case_id']}: {row.get('status')} gate={row.get('gate')} reason={row.get('gate_reason')}",
                flush=True,
            )
        production_solve_results = run_bounded_real_solve_probe(args, plan["bounded_real_solve_probe"])

    payload = {
        "schema_version": 1,
        "kind": "rhs1_pas_matrixfree_probe",
        "plan": plan,
        "results": results,
        "production_solve_results": production_solve_results,
        "summary": summarize_results(results, plan["production_floor_preflight"], production_solve_results),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
