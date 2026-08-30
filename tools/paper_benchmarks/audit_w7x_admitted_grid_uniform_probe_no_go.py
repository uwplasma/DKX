"""Audit the bounded W7-X uniform admitted-grid launch no-go.

This artifact is the one negative result in the registry whose evidence is a
stopped run rather than a comparison: a uniform 17-point all-root search at the
fixed-field-admitted W7-X transport grid was launched, ran for 2551 s at a
21.9 GB peak process footprint, completed zero surfaces, and was stopped. Every
other campaign has an auditor; this one did not, which left a registered claim
whose acceptance rule lived only inside a test.

What is checked here is what makes the record honest rather than a story about a
slow run:

* the launch really was bounded before it started -- the preflight numbers are
  present and internally consistent with the declared hierarchy;
* nothing was completed and nothing was written, so no root, flux, or no-root
  conclusion can be read out of it;
* the route diagnosis explains the stop by memory, with the two rejected
  resident routes each larger than the host's 24 GiB and the executed
  row-on-demand route smaller;
* the outcome refuses both a launch admission and a numerical-failure claim;
* the sealed external input and log are identified by full-length digests.

Usage::

    python tools/paper_benchmarks/audit_w7x_admitted_grid_uniform_probe_no_go.py \
        --artifact validation/w7x_admitted_grid_uniform_probe_no_go_v1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "dkx.w7x_admitted_grid_uniform_probe_no_go.v1"

#: The host the campaign ran on. Both routes the memory guard rejected have to
#: exceed it, or "operationally infeasible" is not what the numbers say.
HOST_MEMORY_BYTES = 24 * 1024**3


def audit(artifact: Path, *, results_root: Path | None = None) -> dict[str, Any]:
    """Return a report whose ``pass`` is true when the no-go record is sound.

    ``results_root`` is accepted for signature parity with the other auditors;
    this artifact references no re-readable NetCDF result, because the run
    deliberately produced none.
    """
    payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
    errors: list[str] = []

    if payload.get("schema") != SCHEMA:
        errors.append(f"schema is {payload.get('schema')!r}, expected {SCHEMA!r}")
    if payload.get("claim_scope") != "bounded_launch_diagnosis_only":
        errors.append(f"claim_scope is {payload.get('claim_scope')!r}")

    case = payload.get("case", {})
    preflight = payload.get("preflight", {})
    measurement = payload.get("measurement", {})
    route = payload.get("route_diagnosis", {})
    outcome = payload.get("outcome", {})
    source = payload.get("source", {})

    # A bounded launch means the work was counted before it started. Two
    # surfaces at 825 evaluations each is the profile ceiling the preflight
    # claims; if those three numbers disagree the run was not actually bounded.
    surfaces = len(case.get("surfaces", ()))
    per_surface = preflight.get("max_evaluations_per_surface")
    profile_total = preflight.get("max_profile_evaluations")
    if surfaces and per_surface is not None and profile_total is not None:
        if per_surface * surfaces != profile_total:
            errors.append(
                f"preflight ceiling is inconsistent: {per_surface} per surface "
                f"x {surfaces} surfaces != {profile_total}"
            )
    hierarchy = preflight.get("hierarchy_points")
    if not isinstance(hierarchy, int) or hierarchy <= case.get("search_points", 0):
        errors.append(
            f"hierarchy_points {hierarchy!r} does not exceed the "
            f"{case.get('search_points')!r} uniform search points"
        )

    # Nothing finished. This is what forbids reading any physics out of the run.
    if measurement.get("completed_surfaces") != 0:
        errors.append(
            f"completed_surfaces is {measurement.get('completed_surfaces')!r}, "
            "expected 0 for a stopped launch"
        )
    if measurement.get("result_written") is not False:
        errors.append("result_written must be false for a stopped launch")
    if measurement.get("termination") != "operator_stopped_after_bounded_no_go":
        errors.append(f"termination is {measurement.get('termination')!r}")
    for name in ("wall_seconds", "maximum_rss_bytes", "peak_process_footprint_bytes"):
        value = measurement.get(name)
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"measurement.{name} is {value!r}")

    # The stop has to be explained by memory, not by an unexplained slowdown.
    reusable = route.get("reusable_dense_coarse_bands_bytes")
    schur = route.get("schur_lu_factors_bytes")
    checkpointed = route.get("checkpointed_dense_factors_per_subsystem_bytes")
    if not isinstance(reusable, (int, float)) or reusable <= HOST_MEMORY_BYTES:
        errors.append(
            f"reusable dense bands {reusable!r} do not exceed the host's "
            f"{HOST_MEMORY_BYTES} bytes, so the guard had no reason to reject them"
        )
    if not isinstance(schur, (int, float)) or schur <= HOST_MEMORY_BYTES:
        errors.append(f"schur LU factors {schur!r} do not exceed host memory")
    if not isinstance(checkpointed, (int, float)) or checkpointed >= HOST_MEMORY_BYTES:
        errors.append(
            f"the executed checkpointed route {checkpointed!r} is not smaller "
            "than host memory, so it is not the route that was actually run"
        )
    if route.get("executed_route") != "solvax_block_thomas_checkpointed_row_on_demand":
        errors.append(f"executed_route is {route.get('executed_route')!r}")

    # The claim boundary. A no-go must not become no-root evidence.
    if outcome.get("uniform_high_grid_launch_admitted") is not False:
        errors.append("uniform_high_grid_launch_admitted must be false")
    if outcome.get("numerical_failure_claimed") is not False:
        errors.append("numerical_failure_claimed must be false")
    if not str(outcome.get("next_route", "")).strip():
        errors.append("a no-go must name the route that replaces it")
    exclusions = payload.get("claim_exclusions", ())
    for required in ("no-root evidence", "ambipolar root or flux result"):
        if required not in exclusions:
            errors.append(f"claim_exclusions must contain {required!r}")

    for name in ("input_sha256", "log_sha256"):
        digest = source.get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"source.{name} is not a full-length digest: {digest!r}")
    if not source.get("external_directory"):
        errors.append("source.external_directory must name where the raw log lives")

    return {
        "schema": "dkx.w7x_admitted_grid_uniform_probe_no_go.audit.v1",
        "artifact": str(artifact),
        "pass": not errors,
        "errors": errors,
        "launch_admitted": outcome.get("uniform_high_grid_launch_admitted"),
        "numerical_failure_claimed": outcome.get("numerical_failure_claimed"),
        "completed_surfaces": measurement.get("completed_surfaces"),
        "peak_process_footprint_bytes": measurement.get("peak_process_footprint_bytes"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("validation/w7x_admitted_grid_uniform_probe_no_go_v1.json"),
    )
    parser.add_argument("--results-root", type=Path, default=None)
    args = parser.parse_args()
    report = audit(args.artifact, results_root=args.results_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
