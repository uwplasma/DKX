#!/usr/bin/env python3
"""Audit the pinned DSHAPE/NCSX/W7-X independent-validation evidence.

The expensive kinetic solves are intentionally not a CI dependency.  This
script recomputes every normalization and acceptance value from the checked
machine-readable artifact, verifies all DKX-owned input and compact-output
checksums, and optionally verifies the pinned YANCC checkout inputs:

    python tools/paper_benchmarks/audit_independent_cross_code_validation.py
    python tools/paper_benchmarks/audit_independent_cross_code_validation.py \
        --yancc-root ../YANCC

The exact inputs, environment, normalization, and measured outputs are
documented in ``validation/README.md`` and its linked artifact.  Keeping the
audit separate from the multi-gigabyte JAX solves makes the arithmetic and
claim boundary reviewable in ordinary CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

# The evidence helpers live in ``tools/release/``, which is checkout tooling
# rather than an installed package, so the repository root has to be importable
# before the import below -- running this file directly puts only its own
# directory on the path.  Under pytest ``tests/conftest.py`` has already done it.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.release.artifacts import (  # noqa: E402
    coefficient_relative_errors,
    dkes_to_beidler,
    nu_prime_for_nu_over_v,
)


DEFAULT_ARTIFACT = ROOT / "validation" / "independent_cross_code_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, float]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reference_conversion(case: dict[str, Any]) -> dict[str, float]:
    raw = case["reference"]["raw_dkes"]
    norm = case["normalization"]
    d33_kwargs = (
        {"d33_spitzer": raw["D33_spitzer"]}
        if "D33_spitzer" in raw
        else {"raw_fsab_b2": raw["fsab_B2"]}
    )
    return dkes_to_beidler(
        d11=raw["D11"],
        d31=raw["D31"],
        d13=raw["D13"],
        d33=raw["D33"],
        nu_over_v=case["equations"]["nu_over_v_per_m"],
        g_hat=norm["g_hat_T_m"],
        iota=norm["iota"],
        b0_over_bbar=norm["B0_T"],
        r_hat=norm["local_r_m"],
        raw_b0_over_bbar=norm["raw_B0_T"],
        cross_orientation=norm["cross_orientation"],
        **d33_kwargs,
    )


def audit(artifact: Path, *, yancc_root: Path | None = None) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    for case in payload["cases"]:
        case_id = str(case["id"])
        inputs = case["inputs"]
        deck = ROOT / inputs["dkx_deck"]
        if not deck.exists() or _sha256(deck) != inputs["dkx_deck_sha256"]:
            errors.append(f"{case_id}: DKX deck checksum mismatch")

        if yancc_root is not None:
            for path_key, checksum_key in (
                ("external_geometry", "external_geometry_sha256"),
                ("external_booz_xform", "external_booz_xform_sha256"),
                ("external_reference_table", "external_reference_table_sha256"),
            ):
                if path_key not in inputs:
                    continue
                rel = str(inputs[path_key]).removeprefix("YANCC/")
                external = yancc_root / rel
                if not external.exists() or _sha256(external) != inputs[checksum_key]:
                    errors.append(f"{case_id}: {path_key} checksum mismatch")

        raw = case["reference"]["raw_dkes"]
        if _canonical_sha256(raw) != case["reference"]["raw_output_sha256"]:
            errors.append(f"{case_id}: compact reference-output checksum mismatch")
        if _canonical_sha256(case["dkx"]["beidler"]) != case["dkx"]["output_sha256"]:
            errors.append(f"{case_id}: compact DKX-output checksum mismatch")

        norm = case["normalization"]
        mapped_nu = nu_prime_for_nu_over_v(
            case["equations"]["nu_over_v_per_m"],
            g_hat=norm["g_hat_T_m"],
            i_hat=norm["i_hat_T_m"],
            iota=norm["iota"],
            b0_over_bbar=norm["B0_T"],
            nu_d_hat_x0=norm["nu_d_hat_x0"],
        )
        if abs(mapped_nu - norm["nu_prime"]) > 2e-14 * abs(norm["nu_prime"]):
            errors.append(f"{case_id}: nuPrime mapping mismatch")

        converted = _reference_conversion(case)
        for key, expected in case["reference"]["beidler"].items():
            if abs(converted[key] - expected) > 2e-14 * abs(expected):
                errors.append(f"{case_id}: {key} normalization mismatch")
        relative = coefficient_relative_errors(case["dkx"]["beidler"], converted)
        for key, expected in case["comparison"]["relative_error"].items():
            arithmetic_slack = max(2e-12 * abs(expected), 5e-16)
            if abs(relative[key] - expected) > arithmetic_slack:
                errors.append(f"{case_id}: {key} relative-error mismatch")
        maximum = max(relative.values())
        passed = maximum <= float(case["comparison"]["relative_tolerance"])
        if passed is not bool(case["comparison"]["pass"]):
            errors.append(f"{case_id}: stored pass flag disagrees with recomputed gate")
        rows.append({"id": case_id, "maximum_relative_error": maximum, "pass": passed})

    all_pass = all(row["pass"] for row in rows)
    if all_pass is not bool(payload["acceptance"]["all_cases_pass"]):
        errors.append("top-level all_cases_pass disagrees with recomputed gates")
    return {
        "schema": "dkx.independent_cross_code.audit.v1",
        "artifact": str(artifact),
        "external_inputs_verified": yancc_root is not None,
        "cases": rows,
        "errors": errors,
        "pass": not errors and all(row["pass"] for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--yancc-root", type=Path)
    args = parser.parse_args()
    report = audit(args.artifact, yancc_root=args.yancc_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
