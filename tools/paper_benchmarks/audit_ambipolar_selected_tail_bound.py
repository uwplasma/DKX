#!/usr/bin/env python3
"""Audit selected-field truncated-route Legendre-tail bound evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation/ambipolar_selected_tail_bound_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(artifact: Path, *, results_root: Path | None = None) -> dict[str, Any]:
    import dkx  # noqa: PLC0415

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    errors: list[str] = []
    source = payload["source"]
    input_record = source["input"]
    input_path = ROOT / input_record["path"]
    if (
        input_path.stat().st_size != input_record["bytes"]
        or _sha256(input_path) != input_record["sha256"]
    ):
        errors.append("input provenance mismatch")
    else:
        case = dkx.Case.from_file(input_path)
        if case.case_id != payload["case"]["case_id"]:
            errors.append("input case ID mismatch")
        if not case.convergence.retain_legendre_tail:
            errors.append("input does not request tail retention")

    expected = np.asarray(
        payload["case"]["selected_tail_bound_by_surface_speed_species"],
        dtype=np.float64,
    )
    if expected.shape != (2, 8, 2):
        errors.append("compact tail-bound shape mismatch")
    if not np.all(np.isfinite(expected)) or not np.all((expected >= 0.0) & (expected <= 1.0)):
        errors.append("compact tail bounds are not finite ratios")
    if float(np.max(expected)) != payload["case"]["maximum_tail_bound"]:
        errors.append("compact maximum-tail arithmetic mismatch")
    if payload["outcome"]["phase_space_converged"]:
        errors.append("tail evidence incorrectly claims phase-space convergence")

    external_verified = results_root is not None
    loaded = []
    if results_root is not None:
        for label in ("cold", "warm"):
            record = source["external_results"][label]
            path = results_root / record["file"]
            if (
                not path.is_file()
                or path.stat().st_size != record["bytes"]
                or _sha256(path) != record["sha256"]
            ):
                errors.append(f"external result provenance mismatch: {label}")
                continue
            result = dkx.Result.load(path)
            loaded.append(result)
            if result.case_id != payload["case"]["case_id"]:
                errors.append(f"external result case mismatch: {label}")
                continue
            if result.metadata["legendre_tail_diagnostic"] != payload["case"][
                "diagnostic_status"
            ]:
                errors.append(f"diagnostic status mismatch: {label}")
            if "evaluation_legendre_tail_relative_l2" in result.arrays:
                errors.append(f"truncated result presents an exact full-state tail: {label}")
            tail = np.asarray(
                result.evaluation_legendre_tail_relative_l2_upper_bound
            )
            selected = []
            for surface_index, field in enumerate(result.electric_field_kV_m):
                index = int(
                    np.nanargmin(
                        np.abs(
                            result.evaluation_electric_field_kV_m[surface_index]
                            - field
                        )
                    )
                )
                selected.append(tail[surface_index, index])
            if not np.array_equal(np.asarray(selected), expected):
                errors.append(f"selected tail bounds mismatch: {label}")
            if np.count_nonzero(np.isfinite(tail)) != payload["case"][
                "finite_tail_values"
            ]:
                errors.append(f"tail sparsity mismatch: {label}")
            replay_count = np.count_nonzero(
                result.evaluation_solver_attempt_reason
                == "selected_tail_diagnostic_replay"
            )
            if replay_count != payload["case"]["diagnostic_replays"]:
                errors.append(f"diagnostic replay count mismatch: {label}")
        if len(loaded) == 2:
            for name in loaded[0].arrays:
                if name == "solve_time_s":
                    continue
                left = np.asarray(loaded[0].arrays[name])
                right = np.asarray(loaded[1].arrays[name])
                equal = (
                    np.array_equal(left, right)
                    if left.dtype.kind in "OUS"
                    else np.array_equal(left, right, equal_nan=True)
                )
                if not equal:
                    errors.append(f"cold/warm scientific array mismatch: {name}")

    return {
        "schema": "dkx.ambipolar_selected_tail_bound.audit.v1",
        "artifact": str(artifact),
        "external_results_verified": external_verified,
        "phase_space_converged": payload["outcome"]["phase_space_converged"],
        "errors": errors,
        "pass": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    report = audit(args.artifact, results_root=args.results_root)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
