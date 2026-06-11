#!/usr/bin/env python
"""Materialize the production stress-test manifest.

The public README figures and parity claims are intentionally generated from
checked benchmark artifacts. This script gathers the current gaps into a single
machine-readable manifest so expensive CPU/GPU reruns can be launched from a
bounded, auditable list rather than from ad hoc case names.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO_ROOT / "benchmarks" / "production_stress_manifest_2026-06-11"
DEFAULT_BENCHMARK_SUMMARY = (
    REPO_ROOT
    / "examples"
    / "publication_figures"
    / "artifacts"
    / "sfincs_jax_fortran_suite_benchmark_summary.json"
)
DEFAULT_PRODUCTION_INPUT_MANIFEST = (
    REPO_ROOT / "benchmarks" / "production_resolution_inputs_2026-05-04" / "manifest.json"
)
DEFAULT_QA_BOOTSTRAP = (
    REPO_ROOT
    / "docs"
    / "_static"
    / "figures"
    / "vmec_jax_finite_beta"
    / "qs_paper_qa_same_resolution_11surface.json"
)
DEFAULT_QH_BOOTSTRAP = (
    REPO_ROOT
    / "docs"
    / "_static"
    / "figures"
    / "vmec_jax_finite_beta"
    / "qs_paper_qh_same_resolution_11surface.json"
)
DEFAULT_QI_EVIDENCE = REPO_ROOT / "docs" / "_static" / "qi_seed_robustness_evidence_manifest.json"
DEFAULT_VMEC_JAX_ROOT = Path("/Users/rogeriojorge/local/vmec_jax")
PRODUCTION_RESOLUTION = {"NTHETA": 25, "NZETA": 51, "NXI": 100, "NX": 4}


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_info(path: Path, *, fixture_stability: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    info: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
        "fixture_stability": fixture_stability,
    }
    if resolved.exists():
        info["size_bytes"] = resolved.stat().st_size
        info["sha256"] = _sha256(resolved)
    return info


def _production_cases_by_name(production_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case["case"]): case
        for case in production_manifest.get("cases", [])
        if isinstance(case, dict) and "case" in case
    }


def _production_case_context(
    case_name: str,
    *,
    production_cases: dict[str, dict[str, Any]],
    production_manifest_path: Path,
) -> dict[str, Any]:
    case = production_cases.get(case_name)
    if case is None:
        return {
            "case": case_name,
            "production_case_found": False,
            "production_input": None,
            "production_resolution": None,
            "run_recommendation": None,
            "total_unknowns_estimate": None,
        }

    size_estimate = case.get("size_estimate", {})
    input_rel = case.get("input")
    input_path = production_manifest_path.parent / str(input_rel) if input_rel else None
    return {
        "case": case_name,
        "production_case_found": True,
        "production_input": str(input_path.resolve()) if input_path is not None else None,
        "production_input_exists": bool(input_path and input_path.exists()),
        "production_resolution": case.get("benchmark_resolution"),
        "original_resolution": case.get("original_resolution"),
        "run_recommendation": (
            size_estimate.get("run_recommendation") if isinstance(size_estimate, dict) else None
        ),
        "total_unknowns_estimate": (
            size_estimate.get("total_unknowns_estimate") if isinstance(size_estimate, dict) else None
        ),
        "source_input": case.get("source_input"),
    }


def _benchmark_floor_gaps(
    benchmark_summary: dict[str, Any],
    *,
    production_cases: dict[str, dict[str, Any]],
    production_manifest_path: Path,
) -> dict[str, list[dict[str, Any]]]:
    metadata = benchmark_summary.get("metadata", {})
    raw_by_backend = metadata.get("resolution_floor_violations", {})
    if not isinstance(raw_by_backend, dict):
        return {}

    gaps: dict[str, list[dict[str, Any]]] = {}
    for backend, rows in raw_by_backend.items():
        if not isinstance(rows, list):
            continue
        gaps[str(backend)] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            case_name = str(row.get("case", ""))
            context = _production_case_context(
                case_name,
                production_cases=production_cases,
                production_manifest_path=production_manifest_path,
            )
            gaps[str(backend)].append(
                {
                    **context,
                    "backend": str(backend),
                    "observed_resolution": row.get("resolution"),
                    "required_resolution": row.get("required"),
                    "violating_fields": row.get("fields"),
                    "reason": row.get("reason"),
                }
            )
    return gaps


def _short_fortran_runtime_cases(
    benchmark_summary: dict[str, Any],
    *,
    production_cases: dict[str, dict[str, Any]],
    production_manifest_path: Path,
) -> list[dict[str, Any]]:
    metadata = benchmark_summary.get("metadata", {})
    rows = metadata.get("excluded_low_fortran_runtime_cases", [])
    if not isinstance(rows, list):
        return []

    result: list[dict[str, Any]] = []
    min_runtime = metadata.get("min_fortran_runtime_s")
    for row in rows:
        if not isinstance(row, dict):
            continue
        case_name = str(row.get("case", ""))
        context = _production_case_context(
            case_name,
            production_cases=production_cases,
            production_manifest_path=production_manifest_path,
        )
        result.append(
            {
                **context,
                "fortran_runtime_s": row.get("fortran_runtime_s"),
                "required_for_public_plot_s": min_runtime,
                "reason": "fortran_runtime_below_public_plot_floor",
            }
        )
    return result


def _bootstrap_entry(case_name: str, path: Path) -> dict[str, Any]:
    payload = _json_load(path)
    metrics = payload.get("metrics", {})
    performance = payload.get("performance", {})
    selected_metrics = {
        key: metrics.get(key)
        for key in (
            "requested_points",
            "completed_points",
            "failed_points",
            "max_jax_relative_difference_vs_fortran_same_resolution",
            "max_jax_relative_difference_vs_redl",
            "max_errorbar_rel_to_baseline",
            "sfincs_jax_elapsed_s_sum",
            "sfincs_fortran_v3_elapsed_s_sum_on_jax_surfaces",
            "sfincs_jax_solver_memory_mb_max",
            "sfincs_fortran_v3_mumps_memory_mb_max_on_jax_surfaces",
        )
    }
    return {
        "case": case_name,
        "source": str(path.resolve()),
        "source_exists": path.exists(),
        "current_claim_boundary": payload.get("claim_boundary"),
        "current_grid_label": "13x13x21x5 diagnostic, 11 radial surfaces",
        "production_target_resolution": PRODUCTION_RESOLUTION,
        "metrics": selected_metrics,
        "performance": performance,
        "next_required_ladder": [
            "rerun same QA/QH surfaces at a bounded mid-grid with residual-clean solves",
            "promote to 25x39x60x7 or documented production-equivalent floor",
            "keep SFINCS_JAX, SFINCS Fortran v3, and Redl on the same radii and normalization",
            "regenerate error bars from real-space and velocity-space refinement",
        ],
    }


def _qi_candidate_specs(vmec_jax_root: Path) -> list[dict[str, str | Path]]:
    build_static = vmec_jax_root / "docs" / "_build" / "html" / "_static"
    docs_static = vmec_jax_root / "docs" / "_static"
    examples_data = vmec_jax_root / "examples" / "data"
    return [
        {
            "label": "qi_nfp1_readme_final",
            "nfp": "1",
            "path": build_static / "qi_readme_cases" / "nfp1" / "wout_final.nc",
            "fixture_stability": "built_docs_ephemeral",
        },
        {
            "label": "qi_nfp2_target_helicity_readme_final",
            "nfp": "2",
            "path": build_static / "qi_readme_cases" / "nfp2_target_helicity" / "wout_final.nc",
            "fixture_stability": "built_docs_ephemeral",
        },
        {
            "label": "qi_nfp3_seed3127_readme_final",
            "nfp": "3",
            "path": build_static / "qi_readme_cases" / "nfp3_seed3127" / "wout_final.nc",
            "fixture_stability": "built_docs_ephemeral",
        },
        {
            "label": "qi_nfp4_minimal_readme_final",
            "nfp": "4",
            "path": build_static / "qi_readme_cases" / "nfp4_minimal" / "wout_final.nc",
            "fixture_stability": "built_docs_ephemeral",
        },
        {
            "label": "qi_readme_best_case_static",
            "nfp": "unknown",
            "path": docs_static / "readme_best_cases" / "qi" / "wout_final.nc",
            "fixture_stability": "local_vmec_jax_docs_static",
        },
        {
            "label": "qi_seed3127_examples_data",
            "nfp": "3",
            "path": examples_data / "wout_QI_stel_seed_3127.nc",
            "fixture_stability": "local_vmec_jax_example_data",
        },
        {
            "label": "qi_nfp3_fixed_resolution_examples_data",
            "nfp": "3",
            "path": examples_data / "wout_nfp3_QI_fixed_resolution_final.nc",
            "fixture_stability": "local_vmec_jax_example_data",
        },
    ]


def _qi_candidates(vmec_jax_root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for spec in _qi_candidate_specs(vmec_jax_root):
        path = Path(spec["path"])
        candidates.append(
            {
                "label": spec["label"],
                "nfp": spec["nfp"],
                "recommended_fixture_action": (
                    "promote to SFINCS_JAX release/external-data fixture with checksum"
                    if spec["fixture_stability"] == "built_docs_ephemeral"
                    else "use as local candidate only until an owned fixture is published"
                ),
                **_path_info(path, fixture_stability=str(spec["fixture_stability"])),
            }
        )
    return candidates


def _qi_evidence_summary(path: Path) -> dict[str, Any]:
    payload = _json_load(path)
    current_evidence = payload.get("current_evidence", {})
    return {
        "source": str(path.resolve()),
        "release_gate": payload.get("release_gate"),
        "release_gate_reason": payload.get("release_gate_reason"),
        "production_target": payload.get("production_target"),
        "open_blockers": payload.get("open_blockers", []),
        "current_evidence": {
            key: current_evidence.get(key)
            for key in (
                "artifact_count",
                "passing_artifact_count",
                "nonpassing_artifact_count",
                "max_checked_per_axis_resolution_fraction",
                "max_checked_total_size_fraction",
                "production_total_size_uncovered_percent",
            )
        }
        if isinstance(current_evidence, dict)
        else {},
    }


def _summary_counts(
    *,
    production_manifest: dict[str, Any],
    benchmark_floor_gaps: dict[str, list[dict[str, Any]]],
    short_fortran_cases: list[dict[str, Any]],
    qi_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    qi_nfps = sorted({str(row["nfp"]) for row in qi_candidates if str(row.get("nfp")) != "unknown"})
    return {
        "production_manifest_cases": production_manifest.get("case_count"),
        "benchmark_floor_gap_rows": {
            backend: len(rows) for backend, rows in benchmark_floor_gaps.items()
        },
        "benchmark_floor_gap_unique_cases": sorted(
            {row["case"] for rows in benchmark_floor_gaps.values() for row in rows}
        ),
        "short_fortran_runtime_case_count": len(short_fortran_cases),
        "qi_candidate_count": len(qi_candidates),
        "qi_nfps_covered": qi_nfps,
        "qi_existing_candidate_count": sum(1 for row in qi_candidates if row.get("exists")),
    }


def build_manifest(
    *,
    benchmark_summary_path: Path = DEFAULT_BENCHMARK_SUMMARY,
    production_manifest_path: Path = DEFAULT_PRODUCTION_INPUT_MANIFEST,
    qa_bootstrap_path: Path = DEFAULT_QA_BOOTSTRAP,
    qh_bootstrap_path: Path = DEFAULT_QH_BOOTSTRAP,
    qi_evidence_path: Path = DEFAULT_QI_EVIDENCE,
    vmec_jax_root: Path = DEFAULT_VMEC_JAX_ROOT,
) -> dict[str, Any]:
    benchmark_summary = _json_load(benchmark_summary_path)
    production_manifest = _json_load(production_manifest_path)
    production_cases = _production_cases_by_name(production_manifest)
    floor_gaps = _benchmark_floor_gaps(
        benchmark_summary,
        production_cases=production_cases,
        production_manifest_path=production_manifest_path,
    )
    short_fortran_cases = _short_fortran_runtime_cases(
        benchmark_summary,
        production_cases=production_cases,
        production_manifest_path=production_manifest_path,
    )
    qi_candidates = _qi_candidates(vmec_jax_root)

    return {
        "schema_version": 1,
        "kind": "sfincs_jax_production_stress_manifest",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "repo": {
            "root": str(REPO_ROOT),
            "commit": _repo_commit(),
        },
        "sources": {
            "benchmark_summary": str(benchmark_summary_path.resolve()),
            "production_input_manifest": str(production_manifest_path.resolve()),
            "qa_bootstrap": str(qa_bootstrap_path.resolve()),
            "qh_bootstrap": str(qh_bootstrap_path.resolve()),
            "qi_evidence": str(qi_evidence_path.resolve()),
            "vmec_jax_root": str(vmec_jax_root.resolve()),
        },
        "public_benchmark_summary": {
            "reported_case_counts": benchmark_summary.get("metadata", {}).get("reported_case_counts"),
            "source_case_counts": benchmark_summary.get("metadata", {}).get("source_case_counts"),
            "min_fortran_runtime_s": benchmark_summary.get("metadata", {}).get("min_fortran_runtime_s"),
            "canonical_row_source": benchmark_summary.get("metadata", {}).get("canonical_row_source"),
        },
        "production_input_manifest": {
            "case_count": production_manifest.get("case_count"),
            "minimum_3d_resolution": production_manifest.get("minimum_3d_resolution"),
            "minimum_tokamak_resolution": production_manifest.get("minimum_tokamak_resolution"),
            "minimum_tokamak_pas_noer_resolution": production_manifest.get(
                "minimum_tokamak_pas_noer_resolution"
            ),
            "target_fortran_min_runtime_s": production_manifest.get(
                "target_fortran_min_runtime_s"
            ),
        },
        "benchmark_floor_gaps": floor_gaps,
        "short_fortran_runtime_cases": short_fortran_cases,
        "qa_qh_bootstrap": [
            _bootstrap_entry("QA", qa_bootstrap_path),
            _bootstrap_entry("QH", qh_bootstrap_path),
        ],
        "qi_evidence": _qi_evidence_summary(qi_evidence_path),
        "qi_candidates": qi_candidates,
        "open_lanes": [
            {
                "lane": "production_floor_example_suite",
                "status": "open",
                "next_gate": "rerun CPU/GPU rows from benchmark_floor_gaps and short_fortran_runtime_cases",
            },
            {
                "lane": "qa_qh_rhsmode1_bootstrap_production",
                "status": "open",
                "next_gate": "residual-clean next-grid ladder before production 25x39x60x7 claim",
            },
            {
                "lane": "lower_memory_production_replacement",
                "status": "open",
                "next_gate": "strict true-residual admission plus RSS/runtime improvement over exact LU",
            },
            {
                "lane": "true_device_qi_and_production_qi_ladders",
                "status": "open",
                "next_gate": "nfp1-4 owned fixture manifest and CPU/GPU production-resolution ladder",
            },
            {
                "lane": "parallelism_and_scaling",
                "status": "open",
                "next_gate": "single-case multi-GPU and multi-CPU scaling on production-sized work",
            },
            {
                "lane": "docs_tests_and_release_claims",
                "status": "open",
                "next_gate": "README/docs plots regenerated only from passing production artifacts",
            },
        ],
        "execution_order": [
            "Review this manifest and verify all source artifact paths/checksums.",
            "Run bounded CPU probes for local/remote production rows without broad all-suite launches.",
            "Run matching office GPU probes with solver traces and memory accounting.",
            "Run QA/QH next-grid bootstrap-current ladder at identical radii for JAX, Fortran v3, and Redl.",
            "Promote or regenerate QI nfp1-4 VMEC-JAX fixtures into owned external-data artifacts.",
            "Only regenerate README/docs runtime, memory, parity, and bootstrap figures from passing artifacts.",
        ],
        "summary_counts": _summary_counts(
            production_manifest=production_manifest,
            benchmark_floor_gaps=floor_gaps,
            short_fortran_cases=short_fortran_cases,
            qi_candidates=qi_candidates,
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--benchmark-summary", type=Path, default=DEFAULT_BENCHMARK_SUMMARY)
    parser.add_argument(
        "--production-manifest",
        type=Path,
        default=DEFAULT_PRODUCTION_INPUT_MANIFEST,
    )
    parser.add_argument("--qa-bootstrap", type=Path, default=DEFAULT_QA_BOOTSTRAP)
    parser.add_argument("--qh-bootstrap", type=Path, default=DEFAULT_QH_BOOTSTRAP)
    parser.add_argument("--qi-evidence", type=Path, default=DEFAULT_QI_EVIDENCE)
    parser.add_argument("--vmec-jax-root", type=Path, default=DEFAULT_VMEC_JAX_ROOT)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a compact summary JSON object after writing the manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_manifest(
        benchmark_summary_path=args.benchmark_summary,
        production_manifest_path=args.production_manifest,
        qa_bootstrap_path=args.qa_bootstrap,
        qh_bootstrap_path=args.qh_bootstrap,
        qi_evidence_path=args.qi_evidence,
        vmec_jax_root=args.vmec_jax_root,
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_path = args.out_root / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps({"manifest": str(out_path), **manifest["summary_counts"]}, sort_keys=True))
    else:
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
