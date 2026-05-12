"""Differentiable vmec_jax -> booz_xform_jax -> sfincs_jax workflow.

This example is deliberately small enough to run as a documentation and CI-adjacent
gate.  It uses vmec_jax provenance for a VMEC ``wout`` object, transforms one flux
surface with booz_xform_jax, evaluates a differentiable sfincs_jax Boozer-spectrum
geometry proxy, checks the gradient against a centered finite difference, and takes a
few scalar optimization steps.

The optimized scalar is a bounded geometry/transport proxy, not a full kinetic solve.
It is the public handoff point for fully JAX-native geometry workflows while the full
VMEC-boundary-to-transport-solve objective remains a larger research lane.

Only the in-memory spectral scaling, ``booz_xform_jax`` call, and
``sfincs_jax`` Boozer-spectrum proxy objective are differentiated here.  VMEC
file reads, fixed-boundary setup, and the SFINCS kinetic transport solve are
outside this example's differentiated graph.  The ``--check-backends`` and
``--summary-json`` modes expose the same machine-readable workflow contract used
by tests and documentation: optional backend checks are shallow, default CI does
not require ``vmec_jax`` or ``booz_xform_jax``, and full kinetic-transport
gradients are an explicit non-claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep this documentation example readable. The full CLI keeps the persistent
# compilation cache enabled by default; this tiny workflow favors clean output.
os.environ.setdefault("SFINCS_JAX_DISABLE_COMPILATION_CACHE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.jax_geometry_adapters import (  # noqa: E402
    boozer_spectrum_geometry_proxy_objective,
    geometry_proxy_workflow_summary,
    optional_jax_geometry_backend_report,
)


def _default_wout_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_path = os.environ.get("SFINCS_JAX_VMEC_JAX_WOUT", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend(
        [
            Path("/Users/rogeriojorge/local/vmec_jax/examples/data/wout_circular_tokamak.nc"),
            _REPO_ROOT / "tests" / "ref" / "wout_w7x_standardConfig.nc",
        ]
    )
    return candidates


def _find_default_wout() -> Path:
    for candidate in _default_wout_candidates():
        if candidate.exists():
            return candidate
    joined = "\n  ".join(str(path) for path in _default_wout_candidates())
    raise FileNotFoundError(
        "No default VMEC wout fixture found. Provide --wout or set "
        "SFINCS_JAX_VMEC_JAX_WOUT. Checked:\n  "
        f"{joined}"
    )


def _print_backend_status() -> None:
    report = optional_jax_geometry_backend_report()
    summary = geometry_proxy_workflow_summary()
    readiness = _synthetic_backend_readiness_gate()
    contract = report["workflow_contract"]
    status = report["backends"]
    print("Optional JAX geometry backend status:")
    for name in ("vmec_jax", "booz_xform_jax"):
        availability = "available" if status[name] else "missing"
        print(f"  {name}: {availability}")
    print("Runnable paths:")
    print("  no optional dependencies: run this script with --check-backends")
    print(
        "  file-backed setup: pass --wout /path/to/wout.nc "
        "or set SFINCS_JAX_VMEC_JAX_WOUT"
    )
    print(
        "  optional in-memory setup: pass --vmec-case circular_tokamak "
        "when vmec_jax is installed"
    )
    print("Differentiability boundary:")
    print("  differentiated: scaled spectral arrays -> booz_xform_jax -> sfincs_jax proxy objective")
    print(
        "  file-backed/setup only: VMEC file I/O, vmec_jax example solves, "
        "and sfincs_jax VMEC file adapters"
    )
    print("  not claimed: full VMEC-boundary-to-SFINCS-transport gradients")
    print("Public workflow contract:")
    print(f"  version: {contract['contract_version']}")
    print("  default CI requires vmec_jax: false")
    print("  default CI requires booz_xform_jax: false")
    print(f"  no-overclaim gate: {contract['no_overclaim_gate']['status']}")
    print(f"  kinetic gradient status: {contract['no_overclaim_gate']['kinetic_gradient_status']}")
    print("Workflow summary:")
    print(
        "  backend-readiness gate: "
        f"{readiness['status']} "
        f"(abs_grad_error={readiness['absolute_error']:.3e}, "
        f"tol={readiness['tolerance']:.3e}, optional_deps=false)"
    )
    print(f"  numerical gradient gate: {summary['numerical_gradient_gate']['status']}")
    print(f"  kinetic transport gradients: {summary['claims']['not_claimed']}")
    print("Machine-readable report:")
    print("  pass --json with --check-backends for backend and gradient-availability metadata")
    print("  pass --summary-json PATH to write reusable workflow provenance JSON")


def _write_summary_json(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_backend_readiness_gate() -> dict[str, object]:
    """Check the local sfincs_jax Boozer-proxy autodiff path without optionals."""
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, 16, endpoint=False, dtype=jnp.float64)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / 5.0, 12, endpoint=False, dtype=jnp.float64)
    base_bmnc = jnp.asarray([1.0, 0.045, -0.021, 0.012, 0.006], dtype=jnp.float64)
    ixm_b = jnp.asarray([0, 1, 1, 2, 3], dtype=jnp.int32)
    ixn_b = jnp.asarray([0, 0, 1, -1, 2], dtype=jnp.int32)
    non_axis = (ixm_b != 0) | (ixn_b != 0)
    scale0 = jnp.asarray(1.0, dtype=jnp.float64)
    fd_step = 1.0e-5
    rtol = 5.0e-4
    atol = 1.0e-8

    def objective(scale: jnp.ndarray) -> jnp.ndarray:
        bmnc = jnp.where(non_axis, base_bmnc * scale, base_bmnc)
        return boozer_spectrum_geometry_proxy_objective(
            bmnc,
            ixm_b,
            ixn_b,
            theta=theta,
            zeta=zeta,
        )

    value, grad = jax.value_and_grad(objective)(scale0)
    finite_difference = (objective(scale0 + fd_step) - objective(scale0 - fd_step)) / (2.0 * fd_step)
    abs_error = abs(float(grad) - float(finite_difference))
    tolerance = atol + rtol * abs(float(finite_difference))
    return {
        "status": "pass" if abs_error <= tolerance else "fail",
        "optional_dependencies_required": False,
        "claim": "sfincs_jax_boozer_proxy_backend_readiness_only",
        "not_claimed": "vmec_jax or booz_xform_jax transform execution",
        "objective": float(value),
        "autodiff_gradient": float(grad),
        "finite_difference_gradient": float(finite_difference),
        "absolute_error": abs_error,
        "tolerance": tolerance,
        "finite_difference_step": fd_step,
        "rtol": rtol,
        "atol": atol,
        "grid_shape": {"n_theta": int(theta.size), "n_zeta": int(zeta.size)},
        "spectrum_modes": int(base_bmnc.size),
    }


def _load_wout_from_vmec_case(args: argparse.Namespace):
    try:
        import vmec_jax as vj
        from vmec_jax.driver import example_paths, run_fixed_boundary, wout_from_fixed_boundary_run
        from vmec_jax.vmec_tomnsp import vmec_angle_grid
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "Running a VMEC example case requires vmec_jax. Install it or use --wout."
        ) from exc

    input_path, _ = example_paths(args.vmec_case)
    cfg, _ = vj.load_input(str(input_path))
    grid = vmec_angle_grid(
        ntheta=int(args.vmec_ntheta),
        nzeta=int(args.vmec_nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )
    run = run_fixed_boundary(
        input_path,
        max_iter=int(args.vmec_max_iter),
        use_initial_guess=True,
        vmec_project=False,
        verbose=bool(args.verbose_vmec),
        grid=grid,
    )
    wout = wout_from_fixed_boundary_run(run, include_fsq=False, fast_bcovar=True)
    return wout, f"vmec_jax fixed-boundary case '{args.vmec_case}'"


def _load_wout_from_file(path: Path):
    try:
        from vmec_jax.wout import read_wout as read_vmec_jax_wout
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("This example requires vmec_jax.wout.read_wout.") from exc
    return read_vmec_jax_wout(path), f"vmec_jax.wout.read_wout('{path}')"


def _build_boozer_context(wout_like, *, mboz: int, nboz: int, surface: float):
    try:
        from booz_xform_jax import Booz_xform
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit("This example requires booz_xform_jax.") from exc

    bx = Booz_xform()
    bx.read_wout_data(wout_like)
    bx.mboz = int(mboz)
    bx.nboz = int(nboz)

    s_in = np.asarray(bx.s_in, dtype=float)
    surface_index = int(np.argmin(np.abs(s_in - float(surface))))
    return bx, surface_index, float(s_in[surface_index])


def _surface_first_arrays(bx) -> dict[str, jnp.ndarray]:
    return {
        "rmnc": jnp.asarray(np.asarray(bx.rmnc).T),
        "zmns": jnp.asarray(np.asarray(bx.zmns).T),
        "lmns": jnp.asarray(np.asarray(bx.lmns).T),
        "bmnc": jnp.asarray(np.asarray(bx.bmnc).T),
        "bsubumnc": jnp.asarray(np.asarray(bx.bsubumnc).T),
        "bsubvmnc": jnp.asarray(np.asarray(bx.bsubvmnc).T),
        "iota": jnp.asarray(np.asarray(bx.iota)),
        "xm_nyq": jnp.asarray(np.asarray(bx.xm_nyq)),
        "xn_nyq": jnp.asarray(np.asarray(bx.xn_nyq)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wout", type=Path, default=None, help="VMEC wout file read through vmec_jax.")
    parser.add_argument(
        "--check-backends",
        action="store_true",
        help="Print optional backend status and the current differentiability boundary, then exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --check-backends, emit backend and gradient-boundary metadata as JSON.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write reusable workflow summary/provenance JSON to this path.",
    )
    parser.add_argument(
        "--vmec-case",
        default=None,
        help="Optional vmec_jax example case to solve first, e.g. circular_tokamak.",
    )
    parser.add_argument("--vmec-max-iter", type=int, default=1)
    parser.add_argument("--vmec-ntheta", type=int, default=16)
    parser.add_argument("--vmec-nzeta", type=int, default=1)
    parser.add_argument("--verbose-vmec", action="store_true")
    parser.add_argument("--surface", type=float, default=0.5)
    parser.add_argument("--mboz", type=int, default=3)
    parser.add_argument("--nboz", type=int, default=3)
    parser.add_argument("--n-theta", type=int, default=12)
    parser.add_argument("--n-zeta", type=int, default=10)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--fd-step", type=float, default=1.0e-4)
    parser.add_argument("--steps", type=int, default=3, help="Scalar gradient-descent steps.")
    parser.add_argument("--learning-rate", type=float, default=5.0)
    args = parser.parse_args()

    if args.check_backends:
        summary = geometry_proxy_workflow_summary()
        summary["backend_readiness_gate"] = _synthetic_backend_readiness_gate()
        if args.summary_json is not None:
            _write_summary_json(args.summary_json, summary)
        if args.json:
            report = optional_jax_geometry_backend_report()
            report["backend_readiness_gate"] = summary["backend_readiness_gate"]
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            _print_backend_status()
        return 0

    if args.vmec_case:
        wout_like, provenance = _load_wout_from_vmec_case(args)
    else:
        wout_path = args.wout if args.wout is not None else _find_default_wout()
        wout_like, provenance = _load_wout_from_file(wout_path)

    bx, surface_index, selected_surface = _build_boozer_context(
        wout_like,
        mboz=args.mboz,
        nboz=args.nboz,
        surface=args.surface,
    )
    arrays = _surface_first_arrays(bx)
    non_axis = (arrays["xm_nyq"] != 0) | (arrays["xn_nyq"] != 0)
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, int(args.n_theta), endpoint=False)
    zeta = jnp.linspace(0.0, 2.0 * jnp.pi / float(bx.nfp), int(args.n_zeta), endpoint=False)

    from booz_xform_jax.jax_api import booz_xform_jax

    def objective(scale: jnp.ndarray) -> jnp.ndarray:
        scaled_bmnc = jnp.where(non_axis[None, :], arrays["bmnc"] * scale, arrays["bmnc"])
        out = booz_xform_jax(
            rmnc=arrays["rmnc"],
            zmns=arrays["zmns"],
            lmns=arrays["lmns"],
            bmnc=scaled_bmnc,
            bsubumnc=arrays["bsubumnc"],
            bsubvmnc=arrays["bsubvmnc"],
            iota=arrays["iota"],
            xm=bx.xm,
            xn=bx.xn,
            xm_nyq=bx.xm_nyq,
            xn_nyq=bx.xn_nyq,
            nfp=int(bx.nfp),
            mboz=int(bx.mboz),
            nboz=int(bx.nboz),
            asym=bool(bx.asym),
            surface_indices=[surface_index],
        )
        return boozer_spectrum_geometry_proxy_objective(
            out["bmnc_b"][0],
            out["ixm_b"],
            out["ixn_b"],
            theta=theta,
            zeta=zeta,
        )

    scale0 = jnp.asarray(float(args.scale))
    value, gradient = jax.value_and_grad(objective)(scale0)
    step = float(args.fd_step)
    finite_difference = (objective(scale0 + step) - objective(scale0 - step)) / (2.0 * step)
    gradient_error = abs(float(gradient - finite_difference))
    summary = geometry_proxy_workflow_summary(
        provenance=provenance,
        requested_surface=float(args.surface),
        selected_surface=selected_surface,
        boozer_resolution={"mboz": int(bx.mboz), "nboz": int(bx.nboz)},
        grid_shape={"n_theta": int(args.n_theta), "n_zeta": int(args.n_zeta)},
        scale=float(scale0),
        proxy_objective=float(value),
        autodiff_gradient=float(gradient),
        finite_difference_gradient=float(finite_difference),
        finite_difference_step=step,
    )

    print("Differentiable geometry workflow:")
    print(f"  VMEC provenance: {provenance}")
    print(f"  Boozer surface: requested s={args.surface:.3f}, selected s={selected_surface:.6f}")
    print(f"  Boozer resolution: mboz={bx.mboz}, nboz={bx.nboz}")
    print(f"  sfincs_jax proxy objective(scale={float(scale0):.6g}) = {float(value):.12e}")
    print(f"  d objective / d scale (JAX) = {float(gradient):.12e}")
    print(f"  d objective / d scale (centered FD) = {float(finite_difference):.12e}")
    print(f"  abs gradient error = {gradient_error:.3e}")
    print(f"  numerical gradient gate: {summary['numerical_gradient_gate']['status']}")
    print("  differentiated graph: scaled spectral arrays -> booz_xform_jax -> sfincs_jax proxy objective")
    print(
        "  outside graph: file I/O, VMEC setup, sfincs_jax VMEC file adapters, "
        "and kinetic transport solve"
    )
    print("  not claimed: full VMEC-boundary-to-SFINCS kinetic transport gradients")

    if args.summary_json is not None:
        _write_summary_json(args.summary_json, summary)
        print(f"  wrote workflow summary JSON: {args.summary_json}")

    scale = scale0
    for k in range(max(0, int(args.steps))):
        loss, grad = jax.value_and_grad(objective)(scale)
        scale = scale - float(args.learning_rate) * grad
        print(f"  step {k + 1:02d}: proxy={float(loss):.12e}, scale={float(scale):.9f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
