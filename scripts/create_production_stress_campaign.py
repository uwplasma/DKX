#!/usr/bin/env python
"""Create exact sharded commands for production stress campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRESS_MANIFEST = REPO_ROOT / "outputs" / "benchmarks" / "production_stress_manifest_2026-06-11" / "manifest.json"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "benchmarks" / "production_stress_manifest_2026-06-11" / "campaign_plan"
DEFAULT_EXAMPLES_ROOT = REPO_ROOT / "outputs" / "benchmarks" / "production_resolution_inputs_2026-05-04" / "inputs"
DEFAULT_PRODUCTION_MANIFEST = (
    REPO_ROOT / "outputs" / "benchmarks" / "production_resolution_inputs_2026-05-04" / "manifest.json"
)


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


def _json_load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _timeout_for_case(total_unknowns: int | None) -> int:
    if total_unknowns is None:
        return 1800
    if total_unknowns <= 150_000:
        return 900
    if total_unknowns <= 800_000:
        return 1800
    if total_unknowns <= 1_600_000:
        return 3600
    return 5400


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _command_path(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except Exception:  # noqa: BLE001
        return str(path)


def _run_command(
    *,
    case: str,
    backend: str,
    examples_root: Path,
    production_manifest: Path,
    fortran_exe: str,
    out_root: str,
    timeout_s: int,
    gpu_device: str,
) -> str:
    base = [
        "python",
        "scripts/run_scaled_example_suite.py",
        "--examples-root",
        _command_path(examples_root),
        "--production-manifest",
        _command_path(production_manifest),
        "--pattern",
        f"^{case}$",
        "--max-run-recommendation",
        "all",
        "--out-root",
        f"{out_root}/{case}",
        "--fortran-exe",
        fortran_exe,
        "--timeout-s",
        str(int(timeout_s)),
        "--max-attempts",
        "1",
        "--jobs",
        "1",
        "--reset-report",
        "--jax-profile-marks",
        "off",
    ]
    prefix = "PYTHONPATH=. "
    if backend == "gpu":
        prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(gpu_device)} JAX_PLATFORM_NAME=gpu PYTHONPATH=. "
    return prefix + _shell_join(base)


def _case_reasons(stress_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for backend, rows in stress_manifest.get("benchmark_floor_gaps", {}).items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            case = str(row["case"])
            item = cases.setdefault(case, {"case": case, "reasons": [], "backends": set(), "context": row})
            item["reasons"].append("below_public_benchmark_resolution_floor")
            item["backends"].add(str(backend))
            item["context"] = row
    for row in stress_manifest.get("short_fortran_runtime_cases", []):
        if not isinstance(row, dict):
            continue
        case = str(row["case"])
        item = cases.setdefault(case, {"case": case, "reasons": [], "backends": set(), "context": row})
        item["reasons"].append("fortran_runtime_below_public_plot_floor")
        item["backends"].update(("cpu", "gpu"))
        item["context"] = row
    return cases


def build_campaign_plan(
    *,
    stress_manifest_path: Path = DEFAULT_STRESS_MANIFEST,
    examples_root: Path = DEFAULT_EXAMPLES_ROOT,
    production_manifest: Path = DEFAULT_PRODUCTION_MANIFEST,
    cpu_out_root: str = "tests/production_stress_cpu_campaign_2026-06-11",
    gpu_out_root: str = "tests/production_stress_gpu_campaign_2026-06-11",
    fortran_exe: str = "/home/rjorge/sfincs/fortran/version3/sfincs",
    gpu_device: str = "0",
) -> dict[str, Any]:
    stress_manifest = _json_load(stress_manifest_path)
    by_case = _case_reasons(stress_manifest)
    cases: list[dict[str, Any]] = []
    for case, item in sorted(by_case.items()):
        context = item["context"]
        total_unknowns = context.get("total_unknowns_estimate")
        total_unknowns_int = int(total_unknowns) if total_unknowns is not None else None
        timeout_s = _timeout_for_case(total_unknowns_int)
        cpu_command = _run_command(
            case=case,
            backend="cpu",
            examples_root=examples_root,
            production_manifest=production_manifest,
            fortran_exe=fortran_exe,
            out_root=cpu_out_root,
            timeout_s=timeout_s,
            gpu_device=gpu_device,
        )
        gpu_command = _run_command(
            case=case,
            backend="gpu",
            examples_root=examples_root,
            production_manifest=production_manifest,
            fortran_exe=fortran_exe,
            out_root=gpu_out_root,
            timeout_s=timeout_s,
            gpu_device=gpu_device,
        )
        cases.append(
            {
                "case": case,
                "reasons": sorted(set(item["reasons"])),
                "backends": sorted(item["backends"]),
                "production_input": context.get("production_input"),
                "production_resolution": context.get("production_resolution"),
                "run_recommendation": context.get("run_recommendation"),
                "total_unknowns_estimate": total_unknowns_int,
                "timeout_s": timeout_s,
                "commands": {"cpu": cpu_command, "gpu": gpu_command},
            }
        )

    return {
        "schema_version": 1,
        "kind": "sfincs_jax_production_stress_campaign_plan",
        "repo": {"root": str(REPO_ROOT), "commit": _repo_commit()},
        "stress_manifest": str(stress_manifest_path.resolve()),
        "examples_root": str(examples_root),
        "production_manifest": str(production_manifest),
        "case_count": len(cases),
        "cases": cases,
        "run_guidance": [
            "Run one case command at a time for first-pass evidence.",
            "Use reference-results-root to reuse completed Fortran references before GPU reruns.",
            "Do not regenerate public README/runtime plots from rows with status other than parity_ok.",
            "Inspect fortran_profile on reference timeout rows to decide whether factor fill or wall-time budget is the blocker.",
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-manifest", type=Path, default=DEFAULT_STRESS_MANIFEST)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--examples-root", type=Path, default=DEFAULT_EXAMPLES_ROOT)
    parser.add_argument("--production-manifest", type=Path, default=DEFAULT_PRODUCTION_MANIFEST)
    parser.add_argument("--cpu-out-root", default="tests/production_stress_cpu_campaign_2026-06-11")
    parser.add_argument("--gpu-out-root", default="tests/production_stress_gpu_campaign_2026-06-11")
    parser.add_argument(
        "--fortran-exe",
        default="/home/rjorge/sfincs/fortran/version3/sfincs",
    )
    parser.add_argument("--gpu-device", default="0")
    parser.add_argument("--json", action="store_true", help="Print compact summary JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    plan = build_campaign_plan(
        stress_manifest_path=args.stress_manifest,
        examples_root=args.examples_root,
        production_manifest=args.production_manifest,
        cpu_out_root=str(args.cpu_out_root),
        gpu_out_root=str(args.gpu_out_root),
        fortran_exe=str(args.fortran_exe),
        gpu_device=str(args.gpu_device),
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    out_path = args.out_root / "campaign_plan.json"
    out_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    commands_path = args.out_root / "commands.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Run selected lines manually; production rows can be long and memory-heavy.",
    ]
    for row in plan["cases"]:
        lines.append("")
        lines.append(f"# {row['case']} timeout_s={row['timeout_s']} reasons={','.join(row['reasons'])}")
        lines.append(f"# CPU: {row['commands']['cpu']}")
        lines.append(f"# GPU: {row['commands']['gpu']}")
    commands_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    commands_path.chmod(0o755)
    if args.json:
        print(json.dumps({"campaign_plan": str(out_path), "case_count": plan["case_count"]}, sort_keys=True))
    else:
        print(f"Wrote {out_path}")
        print(f"Wrote {commands_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
