"""Skip-safe status scaffold for the optional VMEC JAX workflow.

This helper is intentionally lighter than the full
``examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py`` gate. It does not
import ``vmec_jax`` or ``booz_xform_jax``. It reports whether those optional
packages are importable, records the public differentiability contract, and
prints the exact command that runs the file-backed Boozer-spectrum proxy gate
when the optional stack is available.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.jax_geometry_adapters import (  # noqa: E402
    boozer_spectrum_proxy_transport_gradient_gate,
    geometry_proxy_workflow_summary,
    optional_jax_geometry_backend_report,
)


def _command(parts: list[str]) -> str:
    return shlex.join(parts)


def _proxy_gate_command(
    *,
    wout: Path | None,
    proxy_summary_json: Path | None,
    steps: int,
) -> str:
    cmd = [
        "python",
        "examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py",
        "--wout",
        str(wout) if wout is not None else "/path/to/wout_circular_tokamak.nc",
        "--mboz",
        "3",
        "--nboz",
        "3",
        "--surface",
        "0.5",
        "--steps",
        str(int(steps)),
    ]
    if proxy_summary_json is not None:
        cmd.extend(["--summary-json", str(proxy_summary_json)])
    return _command(cmd)


def _synthetic_backend_readiness_gate() -> dict[str, Any]:
    """Run a no-optional-dependency Boozer-proxy transport autodiff check."""
    return boozer_spectrum_proxy_transport_gradient_gate()


def build_status(
    *,
    wout: Path | None = None,
    proxy_summary_json: Path | None = None,
    steps: int = 0,
) -> dict[str, Any]:
    report = optional_jax_geometry_backend_report()
    summary = geometry_proxy_workflow_summary(backend_status=report["backends"])
    missing = sorted(name for name, available in report["backends"].items() if not available)

    return {
        "workflow": summary["workflow"],
        "status": "ready" if not missing else "skipped",
        "skip_reason": None if not missing else f"missing optional backends: {', '.join(missing)}",
        "optional_backends": dict(report["backends"]),
        "default_ci_requires_optional_backends": False,
        "backend_readiness_gate": _synthetic_backend_readiness_gate(),
        "differentiability_contract": {
            "differentiated_graph": list(summary["workflow_contract"]["differentiated_graph"]),
            "outside_differentiated_graph": list(summary["workflow_contract"]["outside_differentiated_graph"]),
            "no_overclaim_gate": dict(summary["no_overclaim_gate"]),
            "not_claimed": summary["claims"]["not_claimed"],
        },
        "commands": {
            "preflight": _command(["python", "examples/optimization/vmec_jax_workflow_status.py", "--json"]),
            "backend_contract": _command(
                [
                    "python",
                    "examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py",
                    "--check-backends",
                    "--json",
                ]
            ),
            "workflow_summary": _command(
                [
                    "python",
                    "examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py",
                    "--check-backends",
                    "--summary-json",
                    "workflow-summary.json",
                ]
            ),
            "proxy_gradient_gate": _proxy_gate_command(
                wout=wout,
                proxy_summary_json=proxy_summary_json,
                steps=steps,
            ),
        },
        "pytest_gates": [
            "python -m pytest tests/test_vmec_jax_workflow.py tests/test_jax_geometry_adapters.py -q",
            "python -m pytest tests/test_optional_ecosystem_gates.py -q",
        ],
    }


def _print_human(status: dict[str, Any]) -> None:
    print(f"VMEC JAX workflow status: {status['status']}")
    if status["skip_reason"]:
        print(f"  reason: {status['skip_reason']}")
    print("Optional backends:")
    for name, available in status["optional_backends"].items():
        print(f"  {name}: {'available' if available else 'missing'}")
    print("Differentiability contract:")
    gate = status["backend_readiness_gate"]
    print("Backend-readiness gate:")
    print(f"  status: {gate['status']}")
    print(f"  optional dependencies required: {str(gate['optional_dependencies_required']).lower()}")
    print(f"  max gradient abs error: {gate['max_gradient_abs_error']:.3e} <= {gate['gradient_tolerance']:.3e}")
    print("  differentiated graph:")
    for stage in status["differentiability_contract"]["differentiated_graph"]:
        print(f"    - {stage}")
    print("  outside differentiated graph:")
    for stage in status["differentiability_contract"]["outside_differentiated_graph"]:
        print(f"    - {stage}")
    print(f"  no-overclaim gate: {status['differentiability_contract']['no_overclaim_gate']['status']}")
    print(f"  not claimed: {status['differentiability_contract']['not_claimed']}")
    print("Commands:")
    for label, command in status["commands"].items():
        print(f"  {label}: {command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wout", type=Path, help="Wout file to embed in the proxy-gradient command.")
    parser.add_argument(
        "--proxy-summary-json",
        type=Path,
        help="Summary JSON path to embed in the proxy-gradient command.",
    )
    parser.add_argument("--steps", type=int, default=0, help="Gradient-descent steps for the generated proxy command.")
    parser.add_argument("--json", action="store_true", help="Print the status payload as JSON.")
    parser.add_argument("--out-json", type=Path, help="Write the status payload to this path.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero when vmec_jax or booz_xform_jax is unavailable.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = build_status(
        wout=args.wout,
        proxy_summary_json=args.proxy_summary_json,
        steps=int(args.steps),
    )
    text = json.dumps(status, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
    else:
        _print_human(status)
    if args.strict and status["status"] != "ready":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
