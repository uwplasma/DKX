"""Differentiable vmec_jax -> booz_xform_jax -> sfincs_jax workflow.

This example is deliberately small enough to run as a documentation and CI-adjacent
gate.  It uses vmec_jax provenance for a VMEC ``wout`` object, transforms one flux
surface with booz_xform_jax, evaluates a differentiable sfincs_jax Boozer-spectrum
geometry proxy, checks the gradient against a centered finite difference, and takes a
few scalar optimization steps.

The optimized scalar is a bounded geometry/transport proxy, not a full kinetic solve.
It is the public handoff point for fully JAX-native geometry workflows while the full
VMEC-boundary-to-transport-solve objective remains a larger research lane.
"""

from __future__ import annotations

import argparse
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

from sfincs_jax.jax_geometry_adapters import boozer_spectrum_geometry_proxy_objective


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

    print("Differentiable geometry workflow:")
    print(f"  VMEC provenance: {provenance}")
    print(f"  Boozer surface: requested s={args.surface:.3f}, selected s={selected_surface:.6f}")
    print(f"  Boozer resolution: mboz={bx.mboz}, nboz={bx.nboz}")
    print(f"  sfincs_jax proxy objective(scale={float(scale0):.6g}) = {float(value):.12e}")
    print(f"  d objective / d scale (JAX) = {float(gradient):.12e}")
    print(f"  d objective / d scale (centered FD) = {float(finite_difference):.12e}")
    print(f"  abs gradient error = {abs(float(gradient - finite_difference)):.3e}")

    scale = scale0
    for k in range(max(0, int(args.steps))):
        loss, grad = jax.value_and_grad(objective)(scale)
        scale = scale - float(args.learning_rate) * grad
        print(f"  step {k + 1:02d}: proxy={float(loss):.12e}, scale={float(scale):.9f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
